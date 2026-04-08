import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_chip(x_max=9, y_max=11, data=[], ancilla=[], x_type=[], title="Qubit Map"):
    """
    Plot the whole chip and optionally highlight data/ancilla qubits.
    x_max=9, y_max=11 applies for the miami quantum computer.
    """

    fig, ax = plt.subplots()

    # Draw connections between ancilla and neighboring data qubits
    ancilla_coords = [(q % 10, q // 10) for q in ancilla]
    data_coords = [(q % 10, q // 10) for q in data]
    
    for ax_x, ax_y in ancilla_coords:
        neighbors = [
        (ax_x - 1, ax_y),
        (ax_x + 1, ax_y),
        (ax_x, ax_y - 1),
        (ax_x, ax_y + 1),
    ]

        for dx, dy in neighbors:
            if (dx, dy) in data_coords:
                ax.plot([ax_x, dx], [ax_y, dy], color="grey", linewidth=2, zorder=0)

    # Draw connections between neighboring data qubits
    for x, y in data_coords:
        neighbors = [
        (x + 1, y + 1),
        (x + 1, y - 1),
    ]

        for xn, yn in neighbors:
            if (xn, yn) in data_coords:
                ax.plot([x, xn], [y, yn], color="black", linewidth=1, zorder=0)

    # Plot qubits with different colors based on their type
    for y in range(y_max + 1):
        for x in range(x_max + 1):
            q = 10 * y + x

            if q in data:
                color = "darkblue"

            elif q in ancilla:
                idx = ancilla.index(q)
                if idx in x_type:
                    color = "seagreen"        # X-ancilla
                else:
                    color = "darkgreen"   # Z-ancilla
            else:
                    color = "cornflowerblue"

            ax.scatter(x, y, s=300, color=color)
            ax.text(x, y, str(q), ha="center", va="center", fontsize=8, color="white")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("off")
    legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Data',
           markerfacecolor='darkblue', markersize=10),

    Line2D([0], [0], marker='o', color='w', label='X ancilla',
           markerfacecolor='seagreen', markersize=10),

    Line2D([0], [0], marker='o', color='w', label='Z ancilla',
           markerfacecolor='darkgreen', markersize=10),

    Line2D([0], [0], marker='o', color='w', label='Unused',
           markerfacecolor='cornflowerblue', markersize=10),
]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def generate_chip_map(distance, corner_qubit, x_max=9, y_max=11, visualization=False):
    """
    Generate CHIP_MAP for a surface code of given distance.

    Assumption:
    - corner_qubit is the TOP corner of the code
    - qubit IDs are in yx format
    """

    y0 = corner_qubit // 10
    x0 = corner_qubit % 10

    data_coords = []
    ancilla_coords = []

    #Generate data qubits based on the corner qubit and distance
    for j in range(distance):
        for i in range(distance):
            x = x0 - i + j
            y = y0 + i + j
            data_coords.append((x, y))

    # Generate ancilla coordinates based on the data qubits
    for i in range(distance - 1):
            for j in range(distance - 1):
                x = x0 - i + j
                y = y0 + i + j + 1
                ancilla_coords.append((x, y))

    for k in range(0, distance - 1, 2):
        x = x0 - k - 1
        y = y0 + k
        ancilla_coords.append((x, y))

    for k in range(0, distance - 1, 2):
        x = x0 - distance + 1 + k
        y = y0 + distance + k
        ancilla_coords.append((x, y))

    for k in range(1, distance - 1, 2):
        x = x0 + k + 1
        y = y0 + k
        ancilla_coords.append((x, y))
    
    for k in range(1, distance - 1, 2):
        x = x0 + distance - 1 - k
        y = y0 + distance + k
        ancilla_coords.append((x, y))

    # Reorders the data qubits to reflect their physical layout (top to bottom, left to right)
    data_coords.sort(key=lambda p: (p[0] + p[1], p[0])) 
    ancilla_coords.sort(key=lambda p: (p[0] + p[1], p[0]))

    data = [10 * y + x for x, y in data_coords]
    ancilla = [10 * y + x for x, y in ancilla_coords]

    # Determine X-type ancillas based on their x-coordinate parity relative to the corner qubit. Only works for d=3, 5,..
    x_type = {i for i, (x, y) in enumerate(ancilla_coords) if x % 2 != x0 % 2}

    chip_map = {
        "data": data,
        "ancilla": ancilla,
        "x_type": x_type,
    }

    if visualization:
        plot_chip(x_max=x_max, y_max=y_max, data=data, ancilla=ancilla, x_type=x_type)

    return chip_map
