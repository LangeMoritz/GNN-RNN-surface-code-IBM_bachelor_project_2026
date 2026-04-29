import torch, time, os
import torch.nn as nn 
from data import Dataset
from args import Args
from utils import GraphConvLayer, TrainingLogger, group, standard_deviation, lfr_per_round
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy


class GRUDecoder(nn.Module):
    """
    A QEC decoder combining a GNN and an RNN.
    """
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        
        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding =  nn.ModuleList([GraphConvLayer(a, b) for a, b in features])
        
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size, num_layers=args.n_layers,
            batch_first=True
        )

    def embed(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map):
        x = self.embed(x, edge_index, edge_attr, batch_labels)
        x = group(x, label_map)
        _, h = self.rnn(x)
        return self.decoder(h[-1]) 

    def train_model(
            self,
            logger: TrainingLogger | None = None,
            save: str | None = None,
            dataset: Dataset | None = None,
            val_dataset=None,
            n_val_batches: int = 10,
            patience: int | None = None,
        ) -> None:
        """
        Train the decoder. If ``val_dataset`` is provided, best-model
        selection switches to validation accuracy and ``patience`` epochs
        of no val-improvement trigger early stopping.
        """
        local_log = isinstance(logger, TrainingLogger)
        best_model = deepcopy(self.state_dict())

        if local_log:
            logger.on_training_begin(self.args)

        self.train()
        if dataset is None:
            dataset = Dataset(self.args)
        optim = torch.optim.AdamW( self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,)
        schedule = lambda epoch: max(0.95 ** epoch, self.args.min_lr / self.args.lr)
        scheduler = LambdaLR(optim, lr_lambda=schedule)
        loss_fn = nn.BCELoss()
        best_metric = 0.0
        epochs_since_improve = 0

        for i in range(1, self.args.n_epochs + 1):
            if local_log:
                logger.on_epoch_begin(i)

            epoch_loss = 0
            epoch_acc = 0
            n_class_0 = 0
            zero, one = [], []
            data_time = 0
            model_time = 0

            for _ in range(self.args.n_batches):
                optim.zero_grad()

                t0 = time.perf_counter()
                x, edge_index, batch_labels, label_map, edge_attr, flips = dataset.generate_batch()
                t1 = time.perf_counter()

                out = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                loss = loss_fn(out, flips.type(torch.float32))
                loss.backward()
                optim.step()

                t2 = time.perf_counter()

                # Statistics
                data_time += t1 - t0
                model_time += t2 - t1
                epoch_loss += loss.item()
                epoch_acc += (torch.sum(torch.round(out) == flips) / torch.numel(flips)).item()
                n_class_0 += torch.sum(flips.squeeze() == 0).item()
                zero.append(out[flips == 0])
                one.append(out[flips == 1])

            zero = torch.hstack(zero).detach().cpu()
            one = torch.hstack(one).detach().cpu()
            zero_mean, zero_std = zero.mean().item(), zero.std().item()
            one_mean, one_std = one.mean().item(), one.std().item()
            epoch_loss /= self.args.n_batches
            epoch_acc /= self.args.n_batches
            noflip = n_class_0 / (torch.numel(flips) * self.args.n_batches)

            # Optional validation pass
            val_acc = None
            val_metrics = None
            if val_dataset is not None:
                val_metrics = self._evaluate(val_dataset, n_val_batches)
                val_acc = val_metrics["acc"]
                self.train()

            metrics = {
                "loss":  epoch_loss,
                "accuracy": epoch_acc,
                "zero_mean": zero_mean,
                "zero_std": zero_std,
                "one_mean": one_mean,
                "one_std": one_std,
                "noflip": noflip,
                "lr": scheduler.get_last_lr()[0],
                "data_time": data_time,
                "model_time": model_time,
                "val_acc": val_acc if val_acc is not None else float("nan"),
            }

            if local_log:
                logger.on_epoch_end(logs=metrics)

            if i % 10 == 0:
                msg = f"Epoch {i}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}"
                if val_metrics is not None:
                    msg += (f", val_acc={val_metrics['acc']:.4f} "
                            f"(c0={val_metrics['acc_0']:.3f}, "
                            f"c1={val_metrics['acc_1']:.3f}, "
                            f"n1/N={val_metrics['n_1'] / (val_metrics['n_0'] + val_metrics['n_1']):.2f})")
                    # LFR/round if we can read T from the val dataset
                    T = getattr(val_dataset, "t", None) or getattr(val_dataset, "T", None)
                    if T and val_acc:
                        msg += f", LFR/rd={lfr_per_round(val_acc, T) * 100:.2f}%"
                print(msg, flush=True)

            # Best-model selection: val accuracy if val provided, else train accuracy
            current_metric = val_acc if val_acc is not None else epoch_acc
            if current_metric > best_metric:
                best_metric = current_metric
                best_model = deepcopy(self.state_dict())
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if patience is not None and epochs_since_improve >= patience:
                print(f"Early stop at epoch {i}: no improvement for {patience} epochs (best={best_metric:.4f}).", flush=True)
                break

            scheduler.step()

        self.load_state_dict(best_model)

        if local_log:
            logger.on_training_end()

        if save:
            os.makedirs("./models", exist_ok=True)
            torch.save(self.state_dict(), f"./models/{save}.pt")

    def _evaluate(self, dataset, n_batches: int) -> dict:
        """Per-class + overall accuracy over ``n_batches`` (no-grad, eval mode).

        Returns a dict with keys: ``acc``, ``acc_0``, ``acc_1``, ``n_0``, ``n_1``.
        """
        self.eval()
        n0, n1, c0, c1 = 0, 0, 0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, ei, lab, lm, ea, flips = dataset.generate_batch()
                out = self.forward(x, ei, ea, lab, lm)
                pred = torch.round(out)
                is0 = flips == 0
                is1 = flips == 1
                n0 += is0.sum().item()
                n1 += is1.sum().item()
                c0 += ((pred == flips) & is0).sum().item()
                c1 += ((pred == flips) & is1).sum().item()
        total = n0 + n1
        return {
            "acc": (c0 + c1) / total if total > 0 else 0.0,
            "acc_0": c0 / n0 if n0 > 0 else 0.0,
            "acc_1": c1 / n1 if n1 > 0 else 0.0,
            "n_0": n0,
            "n_1": n1,
        }

    def test_model(self, dataset: Dataset, n_iter=1000, verbose=True):
        """
        Evaluates the model by feeding n_iter batches to the decoder and 
        calculating the mean and standard deviation of the accuracy. 
        """
        self.eval()
        accuracy_list = torch.zeros(n_iter)
        data_time, model_time = 0, 0
        for i in tqdm(range(n_iter), disable=not verbose):
            t0 = time.perf_counter()
            x, edge_index, batch_labels, label_map, edge_attr, flips = dataset.generate_batch()
            t1 = time.perf_counter() 
            out = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
            t2 = time.perf_counter()
            accuracy_list[i] = torch.sum(torch.round(out) == flips) / torch.numel(flips)
            data_time += t1 - t0
            model_time += t2 - t1
        accuracy = accuracy_list.mean()
        std = standard_deviation(accuracy, n_iter * dataset.batch_size)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, data time = {data_time:.3f}, model time = {model_time:.3f}")
        return accuracy, std
