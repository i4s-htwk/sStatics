
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, CheckButtons

from sstatics.core.system import Pole
from sstatics.core.methods import InfluenceLine


# Biegelinie w(x), Knotenverformung und System
def rotate(origin, points_x, points_z, degree):
    ox, oz = origin
    px = points_x
    pz = points_z

    angle = np.deg2rad(degree)
    # Rotationsformel anwenden, um gedrehte Koordinaten zu berechnen
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (pz - oz)
    qz = oz + np.sin(angle) * (px - ox) + np.cos(angle) * (pz - oz)

    return qx, qz


def plot_lager_x(node, scale):
    rot = -1 * np.rad2deg(node.rotation)
    # Berechnung der Koordinaten des Dreiecks des Lagers
    if node.u == 'fixed' and node.w == 'free':
        rotation = rot
    else:
        rotation = rot - 90
    x1_rotate, z1_rotate = rotate([node.x, node.z],
                                  np.array([node.x, node.x - 1.5 * scale,
                                            node.x - 1.5 * scale, node.x]),
                                  np.array([node.z, node.z - scale,
                                            node.z + scale, node.z]),
                                  rotation)
    # Berechnung der Koordinaten für parallele Linie:
    # Überprüfung der Lagerungsart
    if node.u == 'fixed' and node.w == 'fixed':
        # bei zweiwertiger Lagerung hat das Lager-Symbol keine parallele Linie
        x2_rotate, z2_rotate = [], []
    else:
        # Berechnung der Koordinaten für parallele Linie:
        x2_rotate, z2_rotate = rotate([node.x, node.z],
                                      np.array([node.x - 2 * scale,
                                                node.x - 2 * scale]),
                                      np.array([node.z - scale,
                                                node.z + scale]),
                                      rotation)

    return ([list(x1_rotate),
            list(x2_rotate)],
            [list(z1_rotate),
             list(z2_rotate)])


def plot_dreiwertig(node, scale):
    x1_rotate, z1_rotate = rotate([node.x, node.z],
                                  np.array([node.x, node.x]),
                                  np.array([node.z - scale, node.z + scale]),
                                  node.rotation)
    return [list(x1_rotate)], [list(z1_rotate)]


def get_node_bars(system):
    bars = system.segmented_bars
    all_nodes = [n for bar in bars for n in (bar.node_i, bar.node_j)]
    unique_nodes = []
    for node in all_nodes:
        if all(not node.same_location(n) for n in unique_nodes):
            unique_nodes.append(node)
    return bars, unique_nodes


def get_unique_nodes(bars):
    all_nodes = [n for bar in bars for n in (bar.node_i, bar.node_j)]
    unique_nodes = []
    for node in all_nodes:
        if all(not node.same_location(n) for n in unique_nodes):
            unique_nodes.append(node)
    return bars, unique_nodes


def plot_node(ax, unique_nodes, scale_node, show_label):
    # Zeichnen der Knoten und Beschriftung
    for idx, node in enumerate(unique_nodes):
        offset = scale_node
        if show_label[0]:
            ax.text(node.x + offset, node.z + offset, f'{idx}',
                    ha='right', va='bottom', color='black')

        if node.phi != 'fixed':
            if node.u != 'free' or node.w != 'free':
                x_coods, z_coords = plot_lager_x(node, scale_node)
                for i in range(len(x_coods)):
                    ax.plot(x_coods[i], z_coords[i], color='black',
                            linewidth=2)
        else:
            x_coods, z_coords = plot_dreiwertig(node, scale_node)
            for i in range(len(x_coods)):
                ax.plot(x_coods[i], z_coords[i], color='black',
                        linewidth=2)


# plot Stab
def plot_bar(ax, bar, idx, color, scale_node, show_label):
    # Zeichne Stabnr.
    if show_label[0]:
        draw_bar_marker(ax, bar, idx, color)

    # Berechnung der Richtung des Stabes
    direction = np.array(
        [bar.node_j.x - bar.node_i.x, bar.node_j.z - bar.node_i.z]
    )
    direction_norm = direction / np.linalg.norm(
        direction)  # Normalisierung
    radius = scale_node / 4  # Durchmesser des Kreises in Datenkoordinaten

    # Zeichne Hinge
    if bar.hinge_u_i:
        x1_rotate, z1_rotate = rotate(
            [bar.node_i.x, bar.node_i.z],
            np.array([bar.node_i.x - 1.5 * radius,
                      bar.node_i.x + 1.5 * radius]),
            np.array([bar.node_i.z - 1.5 * radius,
                      bar.node_i.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x1_rotate, z1_rotate, color=color, linewidth=2, zorder=1)

        x2_rotate, z2_rotate = rotate(
            [bar.node_i.x, bar.node_i.z],
            np.array([bar.node_i.x - 1.5 * radius,
                      bar.node_i.x + 1.5 * radius]),
            np.array([bar.node_i.z + 1.5 * radius,
                      bar.node_i.z + 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x2_rotate, z2_rotate, color=color, linewidth=2, zorder=1)

    if bar.hinge_w_i:
        x1_rotate, z1_rotate = rotate(
            [bar.node_i.x, bar.node_i.z],
            np.array([bar.node_i.x, bar.node_i.x]),
            np.array([bar.node_i.z + 1.5 * radius,
                      bar.node_i.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x1_rotate, z1_rotate, color=color, linewidth=2, zorder=1)

        x2_rotate, z2_rotate = rotate(
            [bar.node_i.x, bar.node_i.z],
            np.array([bar.node_i.x + direction_norm[0] * radius,
                      bar.node_i.x + direction_norm[0] * radius]),
            np.array([bar.node_i.z + 1.5 * radius,
                      bar.node_i.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x2_rotate, z2_rotate, color=color, linewidth=2, zorder=1)

    if bar.hinge_phi_i:
        start_circle_x = bar.node_i.x + direction_norm[0] * radius
        start_circle_z = bar.node_i.z + direction_norm[1] * radius
        circle = plt.Circle(
            (start_circle_x, start_circle_z), radius=radius,
            facecolor='white', edgecolor=color, linewidth=1.5, zorder=2)
        ax.add_artist(circle)

    if bar.hinge_u_j:
        x1_rotate, z1_rotate = rotate(
            [bar.node_j.x, bar.node_j.z],
            np.array([bar.node_j.x - 1.5 * radius,
                      bar.node_j.x + 1.5 * radius]),
            np.array([bar.node_j.z - 1.5 * radius,
                      bar.node_j.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x1_rotate, z1_rotate, color=color, linewidth=2, zorder=1)

        x2_rotate, z2_rotate = rotate(
            [bar.node_j.x, bar.node_j.z],
            np.array([bar.node_j.x - 1.5 * radius,
                      bar.node_j.x + 1.5 * radius]),
            np.array([bar.node_j.z + 1.5 * radius,
                      bar.node_j.z + 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x2_rotate, z2_rotate, color=color, linewidth=2, zorder=1)

    if bar.hinge_w_j:
        x1_rotate, z1_rotate = rotate(
            [bar.node_j.x, bar.node_j.z],
            np.array([bar.node_j.x, bar.node_j.x]),
            np.array([bar.node_j.z + 1.5 * radius,
                      bar.node_j.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x1_rotate, z1_rotate, color=color, linewidth=2, zorder=1)

        x2_rotate, z2_rotate = rotate(
            [bar.node_j.x, bar.node_j.z],
            np.array([bar.node_j.x - direction_norm[0] * radius,
                      bar.node_j.x - direction_norm[0] * radius]),
            np.array([bar.node_j.z + 1.5 * radius,
                      bar.node_j.z - 1.5 * radius]),
            -np.rad2deg(bar.inclination))
        ax.plot(x2_rotate, z2_rotate, color=color, linewidth=2, zorder=1)

    if bar.hinge_phi_j:
        end_circle_x = bar.node_j.x - direction_norm[0] * radius
        end_circle_z = bar.node_j.z - direction_norm[1] * radius
        circle = plt.Circle(
            (end_circle_x, end_circle_z), radius=radius,
            facecolor='white', edgecolor=color, linewidth=1.5, zorder=2)
        ax.add_artist(circle)

    # Zeichne Stab
    if (bar.hinge_u_i or bar.hinge_w_i or bar.hinge_phi_i):
        x_coords = [bar.node_i.x + direction_norm[0] * radius, bar.node_j.x]
        z_coords = [bar.node_i.z + direction_norm[1] * radius, bar.node_j.z]
    elif (bar.hinge_u_j or bar.hinge_w_j or bar.hinge_phi_j):
        x_coords = [bar.node_i.x, bar.node_j.x - direction_norm[0] * radius]
        z_coords = [bar.node_i.z, bar.node_j.z - direction_norm[1] * radius]
    elif ((bar.hinge_u_i or bar.hinge_w_i or bar.hinge_phi_i) and
          (bar.hinge_u_j or bar.hinge_w_j or bar.hinge_phi_j)):
        x_coords = [bar.node_i.x + direction_norm[0] * radius,
                    bar.node_j.x - direction_norm[0] * radius]
        z_coords = [bar.node_i.z + direction_norm[1] * radius,
                    bar.node_j.z - direction_norm[1] * radius]
    else:
        x_coords = [bar.node_i.x, bar.node_j.x]
        z_coords = [bar.node_i.z, bar.node_j.z]
    ax.plot(x_coords, z_coords, color=color, linewidth=2, zorder=1)


def draw_bar_marker(ax, bar, idx, color):
    # Berechne den Mittelpunkt des Stabs
    mid_x = (bar.node_i.x + bar.node_j.x) / 2
    mid_z = (bar.node_i.z + bar.node_j.z) / 2

    # Zeichne den Marker für den Stabmittelpunkt
    ax.scatter(mid_x, mid_z, s=220, c='white', edgecolor=color,
               linewidth=1.5, marker='o', zorder=2)

    # Füge den Text für die Stabnummer hinzu
    ax.text(mid_x, mid_z, f'{idx}', ha='center', va='center', color=color,
            zorder=3)


def identify_bars(disp_bars, bars):
    start_idx = end_idx = None
    show_bars = []
    for bar in disp_bars:
        if bar in bars:
            show_bars.append(bar)
            continue
        for divided_bar in bars:
            if (divided_bar.node_i.x == bar.node_i.x and
                    divided_bar.node_i.z == bar.node_i.z):
                start_idx = bars.index(divided_bar)
            if (divided_bar.node_j.x == bar.node_j.x and
                    divided_bar.node_j.z == bar.node_j.z):
                end_idx = bars.index(divided_bar) + 1
                break
        show_bars = show_bars + list(bars[start_idx:end_idx])

    return show_bars


def plot_InfluenceLine(influence_line_obj: InfluenceLine, deform=None,
                       force=None, initial_scale=1, num_points=100,
                       disp_bars=None):
    show_label = [True]
    show_grid_system = [False]
    biegelinie = [True]
    show_values = [True]
    show_solution = [True]
    show_polplan = [False]
    show_divided_bars = [False]

    # input System
    bars_input, unique_nodes_input = get_node_bars(influence_line_obj.system)
    # modified system
    bars, unique_nodes = get_node_bars(influence_line_obj.modified_system)

    show_bars = identify_bars(disp_bars, bars)

    b, show_nodes = get_unique_nodes(show_bars)
    x_solution, z_solution = zip(*[(i.x, i.z) for i in show_nodes])

    x_sys, z_sys = zip(*[(i.x, i.z) for i in unique_nodes])

    z_solution = tuple([0])

    # Maximale Gesamtabmessung
    max_length = max(max(x_sys) - min(x_sys), (max(z_sys) - min(z_sys)) * 2)

    scale_node = 0.04 * max_length
    offset = scale_node * 1.2

    def draw_plot(scale):
        ax.clear()
        # ax_value.clear()
        #
        # relevant_values = []
        #
        # for arr in deform:
        #     relevant_values.extend(arr[[0,1,3,4]].flatten())
        #
        # min_value = min(relevant_values) * scale
        # max_value = max(relevant_values) * scale
        #
        # ylim_value = abs(max((min_value,max_value),key=abs))
        #
        # if show_grid_2[0]:
        #     ax_value.grid(True, color='red',linewidth=0.3, zorder=0)
        # else:
        #     ax_value.grid(False)
        #
        # ax_value.set_ylabel('Einflusslinie',color='red',labelpad=20)
        # ax_value.yaxis.set_label_position('right')
        # ax_value.invert_yaxis()
        #
        # ax_value.spines['right'].set_color('red')
        # ax_value.spines['right'].set_linewidth(2)
        #
        # ax_value.tick_params(axis='y', colors='red')
        #
        # if ylim_value != 0:
        #     ax_value.set_ylim(ylim_value + ylim_value / 10,
        #                       -ylim_value - ylim_value / 10)

        def scientific_formatter(x: float):
            num_str = f"{x:.99g}"
            num_str = num_str.replace(".", "").replace("-", "")
            anzahl = len(num_str)
            if anzahl > 4:
                return f'{x:.2e}'
            else:
                return f'{x}'

        if show_divided_bars[0]:
            plot_node(ax, unique_nodes, scale_node, show_label)
        else:
            plot_node(ax, unique_nodes_input, scale_node, show_label)

        if force is not None:
            title = 'Einflusslinie statisch unbestimmtes System'
            if show_divided_bars[0]:
                for idx, bar in enumerate(bars):
                    color = 'black'
                    plot_bar(ax, bar, idx, color, scale_node, show_label)
            else:
                for idx, bar in enumerate(bars_input):
                    color = 'black'
                    plot_bar(ax, bar, idx, color, scale_node, show_label)

            if show_solution[0]:
                for idx, bar in enumerate(bars):
                    if bar in show_bars:
                        if biegelinie[0]:
                            scale = scale
                            points = bar.deform_line(
                                deform=deform[idx], force=force[idx],
                                scale=scale, n_points=num_points)
                            ax.plot(points[0], points[1], linestyle='-',
                                    color='r')

                            x_1 = [points[0][num_points - 1],
                                   bar.node_j.x]
                            z_1 = [points[1][num_points - 1],
                                   bar.node_j.z]

                            ax.plot(x_1, z_1, linestyle='-', color='r')

                            x_2 = [points[0][0],
                                   bar.node_i.x]
                            z_2 = [points[1][0],
                                   bar.node_i.z]

                            ax.plot(x_2, z_2, linestyle='-', color='r')

                            if show_values[0]:
                                ax.text(points[0][0],
                                        points[1][0] - 0.03 * scale,
                                        scientific_formatter(
                                            deform[idx][1][0]),
                                        fontsize=8, color="red")
                                ax.text(points[0][num_points - 1],
                                        points[1][num_points - 1]
                                        - 0.03 * scale,
                                        scientific_formatter(
                                            deform[idx][4][0]),
                                        fontsize=8, color="red")
                        else:
                            scale = 10000000
                            deform_vals = np.dot(bar.transformation_matrix(
                                to_node_coord=False), deform[idx])
                            x = [bar.node_i.x + deform_vals[0] * scale,
                                 bar.node_j.x + deform_vals[3] * scale]
                            z = [bar.node_i.z + deform_vals[1] * scale,
                                 bar.node_j.z + deform_vals[4] * scale]
                            ax.plot(x, z, linestyle='-', color='r')

                            x_1 = [(bar.node_j.x + deform_vals[3] * scale)[0],
                                   bar.node_j.x]
                            z_1 = [(bar.node_j.z + deform_vals[4] * scale)[0],
                                   bar.node_j.z]
                            ax.plot(x_1, z_1, linestyle='-', color='r')

                            x_2 = [(bar.node_i.x + deform_vals[0] * scale)[0],
                                   bar.node_i.x]
                            z_2 = [(bar.node_i.z + deform_vals[1] * scale)[0],
                                   bar.node_i.z]
                            ax.plot(x_2, z_2, linestyle='-', color='r')

                            if show_values[0]:
                                ax.text(x[0], z[0] - 0.03 * scale,
                                        scientific_formatter(
                                            deform[idx][1][0]),
                                        fontsize=8, color="red")
                                ax.text(x[1], z[1] - 0.03 * scale,
                                        scientific_formatter(
                                            deform[idx][4][0]),
                                        fontsize=8, color="red")
        else:
            title = 'Einflusslinie statisch bestimmtes System'
            chains = influence_line_obj.modified_system.polplan.chains

            if show_polplan[0]:
                col_labels = ['Stabnr.', 'Winkel \u03C6']
                row_labels = []
                table_vals = []
                row_colors = []

                rPol_text = rPol_text_generator(
                    influence_line_obj.modified_system)
                colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1,
                                                                len(chains)))
                for i, chain in enumerate(chains):
                    color = colors[i]
                    bar_indices = []
                    for bar in chain.bars:
                        idx = bars.index(bar)
                        bar_indices.append(idx)
                        plot_bar(ax, bar, idx, color, scale_node, show_label)

                    row_labels.append(f'Scheibe {i}')
                    row_colors.append(color)
                    table_vals.append([", ".join(map(str, bar_indices)),
                                       scientific_formatter(chain.angle)])

                    if chain.stiff:
                        continue

                    for rPol in chain.relative_pole:
                        line_dict = chain.absolute_pole_lines_dict
                        if rPol.x is not None:
                            x, z = rPol.x, rPol.z
                            ax.plot(x, z, 'ro', markersize=2, zorder=10)

                            text = rPol_text[rPol.node]
                            for j, text in enumerate(text):
                                j = j + 1
                                ax.text(x, z - offset * j, f'{text}',
                                        color='black')
                            if line_dict is not False:
                                line = line_dict[rPol.node]
                                if not line[1] is None:
                                    plot_gerade(ax, line, (min(x_sys),
                                                           max(x_sys)))

                        else:
                            line = rPol.line()

                            plot_gerade(ax, line, (min(x_sys), max(x_sys)))

                    if chain.absolute_pole is not None:
                        if chain.absolute_pole.x is not None:
                            x, z = chain.absolute_pole.x, chain.absolute_pole.z
                            ax.plot(x, z, 'o', color='b',
                                    markersize=2, zorder=10)
                            ax.text(x, z - offset, f'({i})', color='black')
                        else:
                            line = chain.absolute_pole.line()

                            plot_gerade(ax, line, (min(x_sys), max(x_sys)))

                    if chain.absolute_pole.is_infinite:
                        line = chain.absolute_pole.line()
                        plot_gerade(ax, line, (min(x_sys), max(x_sys)))

                        x = chain.absolute_pole.node.x
                        z = z_pole_unendlich(line, x, min(z_sys))
                        ax.plot(x, z, 'o', color='b', markersize=0, zorder=10)
                        ax.text(x + offset, z, f'∞ ({i})', color='black')
                        ax.text(x - offset, z - offset,
                                f'\u03C6 = {chain.angle}')

                        for rPol in chain.relative_pole:
                            n = Pole(rPol.node, is_infinite=True,
                                     direction=chain.absolute_pole.direction)
                            line = n.line()
                            plot_gerade(ax, line, (min(x_sys), max(x_sys)))
                            x = n.node.x
                            z = z_pole_unendlich(line, x, min(z_sys))
                            ax.plot(x, z, 'o', color='b', markersize=0,
                                    zorder=10)
                            ax.text(x + offset, z, f'∞ ({i})', color='black')

                # Tabelle wird geplottet
                table = ax.table(cellText=table_vals,
                                 cellLoc='left',
                                 colWidths=[0.1] * 3,
                                 rowLabels=row_labels,
                                 colLabels=col_labels,
                                 rowColours=row_colors,
                                 loc='upper right',
                                 zorder=10)
                table.auto_set_font_size(False)
            else:
                if show_divided_bars[0]:
                    for idx, bar in enumerate(bars):
                        color = 'black'
                        plot_bar(ax, bar, idx, color, scale_node, show_label)
                else:
                    for idx, bar in enumerate(bars_input):
                        color = 'black'
                        plot_bar(ax, bar, idx, color, scale_node, show_label)
            if show_solution[0]:
                for idx, bar in enumerate(bars):
                    deform_vals = np.dot(bar.transformation_matrix(
                        to_node_coord=False), deform[idx])
                    x = [bar.node_i.x + deform_vals[0] * scale,
                         bar.node_j.x + deform_vals[3] * scale]
                    z = [bar.node_i.z + deform_vals[1] * scale,
                         bar.node_j.z + deform_vals[4] * scale]
                    ax.plot(x, z, linestyle='-', color='r')

                    x_1 = [(bar.node_j.x + deform_vals[3] * scale)[0],
                           bar.node_j.x]
                    z_1 = [(bar.node_j.z + deform_vals[4] * scale)[0],
                           bar.node_j.z]
                    ax.plot(x_1, z_1, linestyle='-', color='r')

                    x_2 = [(bar.node_i.x + deform_vals[0] * scale)[0],
                           bar.node_i.x]
                    z_2 = [(bar.node_i.z + deform_vals[1] * scale)[0],
                           bar.node_i.z]
                    ax.plot(x_2, z_2, linestyle='-', color='r')

                    if show_values[0]:
                        ax.text(x[0], z[0] - 0.05 * scale,
                                f'{deform[idx][1][0]:.2f}',
                                fontsize=8, color="red")
                        ax.text(x[1], z[1] - 0.05 * scale,
                                f'{deform[idx][4][0]:.2f}',
                                fontsize=8, color="red")

        ax.set_title(title)
        ax.set_xlabel("x-Koordinaten")
        ax.set_ylabel("z-Koordinaten")
        ax.grid(show_grid_system[0])
        ax.axis("equal")
        ax.invert_yaxis()

        # # # Gleiche Achsenlimits setzen
        ax.set_ylim(max(z_solution) + offset, (min(z_solution) - offset))
        ax.set_xlim(min(x_sys) - offset * 2, max(x_sys) + offset * 2)

        fig.canvas.draw_idle()  # Aktualisiert den Plot

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))
    # ax_value = ax.twinx()
    plt.subplots_adjust(left=0.4)
    draw_plot(initial_scale)

    left = 0.05
    width = 0.2

    checkbox_value_ax = plt.axes((left, 0.55, width, 0.3))

    if force is None:
        check_buttons_value = CheckButtons(checkbox_value_ax,
                                           ["Ergebnis anzeigen",
                                            "Werte anzeigen",
                                            "Polplan anzeigen"],
                                           [True, True, False])
    else:
        check_buttons_value = CheckButtons(checkbox_value_ax,
                                           ["Ergebnis anzeigen",
                                            "Werte anzeigen",
                                            "Modifiziertes System anzeigen",
                                            "Analytische Biegelinie"],
                                           [True, True, False, True])

    # TODO: Optionen
    ''' Skalierung '''
    # fig.text(left, 0.5, "Optionen:", fontsize=12, fontweight='bold')
    # # Slider for scaling
    slider_ax = plt.axes((left + 0.06, 0.8, width - 0.07, 0.03))
    scale_slider = Slider(ax=slider_ax, label="Skalierung", valmin=1,
                          valmax=1000, valinit=initial_scale)
    # scale_slider.label.set_visible(False)
    scale_slider.on_changed(lambda val: draw_plot(scale_slider.val))

    ''' Werte anzeigen, Raster an/aus, Beschriftung'''
    # Checkbuttons for toggling deformation and values
    checkbox_ax = plt.axes((left, 0.1, width, 0.3))

    check_buttons = CheckButtons(checkbox_ax,
                                 ["Gitternetzlinien - System",
                                  # "Gitternetzlinien - Einflusslinie",
                                  "Beschriftung"],
                                 [False, True])

    def toggle_display(label):
        if label == "Gitternetzlinien - System":
            show_grid_system[0] = not show_grid_system[0]
        # if label == "Gitternetzlinien - Einflusslinie":
        #     show_grid_2[0] = not show_grid_2[0]
        if label == "Beschriftung":
            show_label[0] = not show_label[0]
        draw_plot(scale_slider.val)

    check_buttons.on_clicked(toggle_display)

    def toggle_display_1(label):
        if label == "Werte anzeigen":
            show_values[0] = not show_values[0]
        if label == "Ergebnis anzeigen":
            show_solution[0] = not show_solution[0]
        if label == "Polplan anzeigen":
            show_polplan[0] = not show_polplan[0]
            show_divided_bars[0] = not show_divided_bars[0]
        if label == "Modifiziertes System anzeigen":
            show_divided_bars[0] = not show_divided_bars[0]
        if label == "Analytische Biegelinie":
            biegelinie[0] = not biegelinie[0]
        draw_plot(scale_slider.val)

    check_buttons_value.on_clicked(toggle_display_1)

    plt.show()


# Plot Verschiebungsplan
def plot_chains(system):
    # if not system.polplan.solved:
    #     return plot(title='System', system=system)
    ax = plt
    bars = system.segmented_bars
    all_nodes = [n for bar in bars for n in (bar.node_i, bar.node_j)]
    unique_nodes = []
    for node in all_nodes:
        if any([node.same_location(n) for n in unique_nodes]) is False:
            unique_nodes.append(node)

    x_sys, z_sys = zip(*[(i.x, i.z) for i in unique_nodes])
    # Maximale Gesamtabmessung
    max_length = max(max(x_sys) - min(x_sys),
                     (max(z_sys) - min(z_sys)) * 2)

    scale_node = 0.04 * max_length

    plt.figure(figsize=(8, 8))

    offset = scale_node * 1.2

    # Zeichnen der Knoten und Beschriftung
    for idx, node in enumerate(unique_nodes):
        offset = scale_node * 1.2

        if node.phi != 'fixed':
            if node.u != 'free' or node.w != 'free':
                x_coods, z_coords = plot_lager_x(node, scale_node)
                for i in range(len(x_coods)):
                    plt.plot(x_coods[i], z_coords[i], color='black',
                             linewidth=2)
        else:
            x_coods, z_coords = plot_dreiwertig(node, scale_node)
            for i in range(len(x_coods)):
                plt.plot(x_coods[i], z_coords[i], color='black', linewidth=2)

    chains = system.polplan.chains

    legend_labels = []  # Für die Legende
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(chains)))

    # Zeichnen der Stäbe und Pfeile
    # for slice_index, scheiben_bars in scheiben.items():
    for i, chain in enumerate(chains):
        # Wähle die Farbe für diese Scheibe
        color = colors[i]
        bar_indices = []
        for bar in chain.bars:
            idx = bars.index(bar)
            bar_indices.append(idx)
            # plot_bar(plt, bar, idx, color, scale_node, [True])
            x_coords = [bar.node_i.x, bar.node_j.x]
            z_coords = [bar.node_i.z, bar.node_j.z]
            plt.plot(x_coords, z_coords, color=color,
                     linewidth=2, zorder=1)  # Stäbe in der jeweiligen Farbe
            # Pfeil in der Mitte des Stabs zeichnen
            # plt.arrow(bar.node_i.x, bar.node_i.z,
            #           bar.node_j.x - bar.node_i.x,
            #           bar.node_j.z - bar.node_i.z,
            #           head_width=scale_node, head_length=scale_node,
            #           fc=color, ec=color,
            #           length_includes_head=True, zorder=2)

            # # Position für die Stabnummer berechnen (Mittelpunkt des Stabs)
            # mid_x = (bar.node_i.x + bar.node_j.x) / 2
            # mid_z = (bar.node_i.z + bar.node_j.z) / 2
            # circle = plt.Circle(
            #     (mid_x,mid_z),radius=scale_node / 2,
            #     facecolor='white',edgecolor=color,linewidth=1.5,zorder=3
            # )
            # plt.gca().add_artist(circle)
            #
            # plt.text(mid_x,mid_z,f'S{idx}',ha='center',va='center',
            #          color=color,zorder=3)  # Stabnummern

            draw_bar_marker(ax, bar, idx, color)

            # Berechnung der Richtung des Stabes
            direction = np.array(
                [bar.node_j.x - bar.node_i.x, bar.node_j.z - bar.node_i.z]
            )
            direction_norm = direction / np.linalg.norm(
                direction)  # Normalisierung
            radius = scale_node / 4

            # Kreise am Stabanfang zeichnen, wenn hinge_phi_i = True
            if bar.hinge_phi_i:
                start_circle_x = bar.node_i.x + direction_norm[0] * radius
                start_circle_z = bar.node_i.z + direction_norm[1] * radius
                circle = plt.Circle(
                    (start_circle_x, start_circle_z), radius=radius,
                    facecolor='white', edgecolor=color, linewidth=1.5, zorder=5
                )
                plt.gca().add_artist(circle)

            # Kreise am Stabende zeichnen, wenn hinge_phi_j = True
            if bar.hinge_phi_j:
                end_circle_x = bar.node_j.x - direction_norm[0] * radius
                end_circle_z = bar.node_j.z - direction_norm[1] * radius
                circle = plt.Circle(
                    (end_circle_x, end_circle_z), radius=radius,
                    facecolor='white', edgecolor=color, linewidth=1.5, zorder=4
                )
                plt.gca().add_artist(circle)

            if bar.hinge_w_i:
                x1_rotate, z1_rotate = rotate(
                    [bar.node_i.x, bar.node_i.z],
                    np.array([bar.node_i.x, bar.node_i.x]),
                    np.array([bar.node_i.z + 1.5 * radius,
                              bar.node_i.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x1_rotate, z1_rotate, color=color,
                         linewidth=2, zorder=4)

                x2_rotate, z2_rotate = rotate(
                    [bar.node_i.x, bar.node_i.z],
                    np.array([bar.node_i.x + direction_norm[0] * radius,
                              bar.node_i.x + direction_norm[0] * radius]),
                    np.array([bar.node_i.z + 1.5 * radius,
                              bar.node_i.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x2_rotate, z2_rotate, color=color,
                         linewidth=2, zorder=4)

            if bar.hinge_w_j:
                x1_rotate, z1_rotate = rotate(
                    [bar.node_j.x, bar.node_j.z],
                    np.array([bar.node_j.x, bar.node_j.x]),
                    np.array([bar.node_j.z + 1.5 * radius,
                              bar.node_j.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x1_rotate, z1_rotate, color=color,
                         linewidth=2, zorder=4)

                x2_rotate, z2_rotate = rotate(
                    [bar.node_j.x, bar.node_j.z],
                    np.array([bar.node_j.x - direction_norm[0] * radius,
                              bar.node_j.x - direction_norm[0] * radius]),
                    np.array([bar.node_j.z + 1.5 * radius,
                              bar.node_j.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x2_rotate, z2_rotate, color=color,
                         linewidth=2, zorder=4)

            if bar.hinge_u_i:
                x1_rotate, z1_rotate = rotate(
                    [bar.node_i.x, bar.node_i.z],
                    np.array([bar.node_i.x - 1.5 * radius,
                              bar.node_i.x + 1.5 * radius]),
                    np.array([bar.node_i.z - 1.5 * radius,
                              bar.node_i.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x1_rotate, z1_rotate, color=color,
                         linewidth=2, zorder=4)

                x2_rotate, z2_rotate = rotate(
                    [bar.node_i.x, bar.node_i.z],
                    np.array([bar.node_i.x - 1.5 * radius,
                              bar.node_i.x + 1.5 * radius]),
                    np.array([bar.node_i.z + 1.5 * radius,
                              bar.node_i.z + 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x2_rotate, z2_rotate, color=color,
                         linewidth=2, zorder=4)

            if bar.hinge_u_j:
                x1_rotate, z1_rotate = rotate(
                    [bar.node_j.x, bar.node_j.z],
                    np.array([bar.node_j.x - 1.5 * radius,
                              bar.node_j.x + 1.5 * radius]),
                    np.array([bar.node_j.z - 1.5 * radius,
                              bar.node_j.z - 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x1_rotate, z1_rotate, color=color,
                         linewidth=2, zorder=4)

                x2_rotate, z2_rotate = rotate(
                    [bar.node_j.x, bar.node_j.z],
                    np.array([bar.node_j.x - 1.5 * radius,
                              bar.node_j.x + 1.5 * radius]),
                    np.array([bar.node_j.z + 1.5 * radius,
                              bar.node_j.z + 1.5 * radius]),
                    -np.rad2deg(bar.inclination))
                plt.plot(x2_rotate, z2_rotate, color=color,
                         linewidth=2, zorder=4)

        if chain.stiff:
            continue

        for rPol in chain.relative_pole:
            line_dict = chain.absolute_pole_lines_dict
            if rPol.x is not None:
                x, z = rPol.x, rPol.z
                plt.plot(x, z, 'ro', markersize=2, zorder=10)

                # text = rPol_text[rPol.node]
                # for j, text in enumerate(text):
                #     j = j + 1
                #     plt.text(x, z - offset * j, f'{text}',color='black')
                if line_dict is not False:
                    line = line_dict[rPol.node]
                    if not line[1] is None:
                        plot_gerade(ax, line, (min(x_sys), max(x_sys)))

            else:
                line = rPol.line()

                plot_gerade(ax, line, (min(x_sys), max(x_sys)))

        if chain.absolute_pole is not None:
            if chain.absolute_pole.x is not None:
                x, z = chain.absolute_pole.x, chain.absolute_pole.z
                plt.plot(x, z, 'o', color='b', markersize=2, zorder=10)
                plt.text(x + offset, z, f'({i})', color='black')
            else:
                line = chain.absolute_pole.line()

                plot_gerade(ax, line, (min(x_sys), max(x_sys)))

        if chain.absolute_pole.is_infinite:
            line = chain.absolute_pole.line()
            plot_gerade(ax, line, (min(x_sys), max(x_sys)))

            x = chain.absolute_pole.node.x
            z = z_pole_unendlich(line, x, min(z_sys))
            plt.plot(x, z, 'o', color='b', markersize=0, zorder=10)
            plt.text(x + offset, z, f'∞ ({i})', color='black')

            for rPol in chain.relative_pole:
                n = Pole(rPol.node, is_infinite=True,
                         direction=chain.absolute_pole.direction)
                line = n.line()
                plot_gerade(ax, line, (min(x_sys), max(x_sys)))
                x = n.node.x
                z = z_pole_unendlich(line, x, min(z_sys))
                plt.plot(x, z, 'o', color='b', markersize=0, zorder=10)
                plt.text(x + offset, z, f'∞ ({i})', color='black')

        # Legende hinzufügen
        legend_labels.append(f'Scheibe {i}: {bar_indices}')

    deform = system.polplan.get_displacement_figure()

    for idx, bar in enumerate(bars):
        scale = 1
        deform_vals = np.dot(bar.transformation_matrix(
            to_node_coord=False), deform[idx])
        x = [bar.node_i.x + deform_vals[0] * scale,
             bar.node_j.x + deform_vals[3] * scale]
        z = [bar.node_i.z + deform_vals[1] * scale,
             bar.node_j.z + deform_vals[4] * scale]
        ax.plot(x, z, linestyle='-', color='r')

    # Legende erstellen
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    plt.legend(handles, legend_labels, title="Scheiben")

    plt.xlabel("X-Koordinate")
    plt.ylabel("Z-Koordinate")
    plt.title("Verschiebungsplan")
    plt.grid(True)
    plt.axis("equal")
    plt.gca().invert_yaxis()  # Z-Achse umdrehen
    plt.show()


# Polplan
def plot_gerade(ax, line, xlim, color='b'):
    """
    Plottet eine Gerade basierend auf ihrer Steigung m und ihrem
    y-Achsenabschnitt n.
    Berücksichtigt auch vertikale und horizontale Geraden.
    """
    m, n = line[0], line[1]
    x_vals = np.linspace(xlim[0], xlim[1], 100)

    if m is None:
        # Vertikale Gerade x = n
        ax.axvline(x=n, color=color, linestyle='--', linewidth=0.5)
    elif m == 0:
        # Horizontale Gerade y = n
        ax.axhline(y=n, color=color, linestyle='--', linewidth=0.5)
    else:
        # Allgemeine Gerade y = m*x + n
        y_vals = m * x_vals + n
        ax.plot(x_vals, y_vals, color=color, linestyle='--', linewidth=0.5)


def z_pole_unendlich(line, x, maximum):
    """
    Plottet eine Gerade basierend auf ihrer Steigung m und ihrem
    y-Achsenabschnitt n.
    Berücksichtigt auch vertikale und horizontale Geraden.
    """
    m, n = line[0], line[1]

    if m is None:
        # Vertikale Gerade x = n
        return maximum * 1.2
    elif m == 0:
        # Horizontale Gerade y = n
        return line[1]
    else:
        # Allgemeine Gerade y = m*x + n
        return m * x + n


def rPol_text_generator(system):
    rPole_register = system.polplan.node_to_chain_map
    rPol_text = {}
    for rPole, chains in rPole_register.items():
        index = []
        text_list = []
        for chain in chains:
            index.append(system.polplan.chains.index(chain))

        for i, j in combinations(index, 2):
            text = f'({i}|{j})'
            text_list.append(text)

        rPol_text[rPole] = text_list
    return rPol_text


def plot(title, system, deform=None, force=None, scale=10, num_points=100):
    bars = system.segmented_bars
    all_nodes = [n for bar in bars for n in (bar.node_i, bar.node_j)]
    unique_nodes = []
    for node in all_nodes:
        if any([node.same_location(n) for n in unique_nodes]) is False:
            unique_nodes.append(node)

    x, z = zip(*[(i.x, i.z) for i in unique_nodes])
    # Maximale Gesamtabmessung
    max_length = max(max(x) - min(x),
                     (max(z) - min(z)) * 2)

    scale_node = 0.04 * max_length

    plt.figure(figsize=(8, 8))

    # Zeichnen der Knoten und Beschriftung
    for idx, node in enumerate(unique_nodes):
        offset = scale_node
        plt.text(node.x + offset, node.z + offset, f'K{idx}', ha='right',
                 va='bottom', color='black')  # Knotennummern

        if node.phi != 'fixed':
            if node.u != 'free' or node.w != 'free':
                x_coods, z_coords = plot_lager_x(node, scale_node)
                for i in range(len(x_coods)):
                    plt.plot(x_coods[i], z_coords[i], color='black',
                             linewidth=2)
        else:
            x_coods, z_coords = plot_dreiwertig(node, scale_node)
            for i in range(len(x_coods)):
                plt.plot(x_coods[i], z_coords[i], color='black', linewidth=2)

    for idx, bar in enumerate(bars):
        x_coords = [bar.node_i.x, bar.node_j.x]
        z_coords = [bar.node_i.z, bar.node_j.z]
        plt.plot(x_coords, z_coords, color='black',
                 linewidth=2)  # Stäbe in der jeweiligen Farbe

        # Position für die Stabnummer berechnen (Mittelpunkt des Stabs)
        mid_x = (bar.node_i.x + bar.node_j.x) / 2
        mid_z = (bar.node_i.z + bar.node_j.z) / 2
        plt.text(mid_x, mid_z, f'S{idx}', ha='center', va='center',
                 color='red')  # Stabnummern

        # Berechnung der Richtung des Stabes
        direction = np.array(
            [bar.node_j.x - bar.node_i.x, bar.node_j.z - bar.node_i.z]
        )
        direction_norm = direction / np.linalg.norm(
            direction)  # Normalisierung
        radius = scale_node / 4  # Durchmesser des Kreises in Datenkoordinaten

        # Kreise am Stabende zeichnen, wenn hinge_phi_j = True
        if bar.hinge_phi_j:
            end_circle_x = bar.node_j.x - direction_norm[0] * radius
            end_circle_z = bar.node_j.z - direction_norm[1] * radius
            circle = plt.Circle(
                (end_circle_x, end_circle_z), radius=radius,
                facecolor='white', edgecolor='black', linewidth=1.5
            )
            plt.gca().add_artist(circle)

        # Kreise am Stabanfang zeichnen, wenn hinge_phi_i = True
        if bar.hinge_phi_i:
            start_circle_x = bar.node_i.x + direction_norm[0] * radius
            start_circle_z = bar.node_i.z + direction_norm[1] * radius
            circle = plt.Circle(
                (start_circle_x, start_circle_z), radius=radius,
                facecolor='white', edgecolor='black', linewidth=1.5
            )
            plt.gca().add_artist(circle)

        if title == 'Biegelinie':
            points = bar.deform_line(deform=deform[idx], force=force[idx],
                                     scale=scale, n_points=num_points)

            plt.plot(points[0], points[1], linestyle='-', color='r')

            print(points[0])
            print(points[1])

        elif title == 'Knotenverformung':
            deform = (bar.transformation_matrix(to_node_coord=False) *
                      deform[idx])
            x = [bar.node_i.x + deform[0] * scale,
                 bar.node_j.x + deform[3] * scale]
            z = [bar.node_i.z + deform[1] * scale,
                 bar.node_j.z + deform[4] * scale]
            plt.plot(x, z, linestyle='-', color='r')

    plt.xlabel("X-Koordinate")
    plt.ylabel("Z-Koordinate")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.gca().invert_yaxis()  # Z-Achse umdrehen
    plt.show()
