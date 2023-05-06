import argparse
import pathlib
import pickle
from typing import Optional, Sequence

import math
import xml.dom.minidom
from itertools import islice
from xml.dom.minidom import parse

import networkx as nx
import numpy as np

from EON_GYM.utils import (
    Modulation,
    Path,
    get_best_modulation_format,
    get_k_shortest_paths,
    get_path_weight,
)

modulations: Optional[Sequence[Modulation]] = None

modulations = (

    Modulation(
        name="BPSK",
        maximum_length=100_000,
        spectral_efficiency=1,
        minimum_osnr=12.6,
        inband_xt=-14,
    ),
    Modulation(
        name="QPSK",
        maximum_length=2_000,
        spectral_efficiency=2,
        minimum_osnr=12.6,
        inband_xt=-17,
    ),
    Modulation(
        name="8QAM",
        maximum_length=1_000,
        spectral_efficiency=3,
        minimum_osnr=18.6,
        inband_xt=-20,
    ),
    Modulation(
        name="16QAM",
        maximum_length=500,
        spectral_efficiency=4,
        minimum_osnr=22.4,
        inband_xt=-23,
    ),
    Modulation(
        name="32QAM",
        maximum_length=250,
        spectral_efficiency=5,
        minimum_osnr=26.4,
        inband_xt=-26,
    ),
    Modulation(
        name="64QAM",
        maximum_length=125,
        spectral_efficiency=6,
        minimum_osnr=30.4,
        inband_xt=-29,
    ),
)

def calculate_geographical_distance(latlong1, latlong2):
    R = 6373.0

    lat1 = math.radians(latlong1[0])
    lon1 = math.radians(latlong1[1])
    lat2 = math.radians(latlong2[0])
    lon2 = math.radians(latlong2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    length = R * c
    return length


def read_sndlib_topology(file):
    graph = nx.Graph()

    with open(file) as file:
        tree = xml.dom.minidom.parse(file)
        document = tree.documentElement

        graph.graph["coordinatesType"] = document.getElementsByTagName("nodes")[
            0
        ].getAttribute("coordinatesType")

        nodes = document.getElementsByTagName("node")
        for node in nodes:
            x = node.getElementsByTagName("x")[0]
            y = node.getElementsByTagName("y")[0]
            # print(node['id'], x.string, y.string)
            graph.add_node(
                node.getAttribute("id"),
                pos=((float(x.childNodes[0].data), float(y.childNodes[0].data))),
            )
        # print('Total nodes: ', graph.number_of_nodes())
        links = document.getElementsByTagName("link")
        for idx, link in enumerate(links):
            source = link.getElementsByTagName("source")[0]
            target = link.getElementsByTagName("target")[0]

            if graph.graph["coordinatesType"] == "geographical":
                length = np.around(
                    calculate_geographical_distance(
                        graph.nodes[source.childNodes[0].data]["pos"],
                        graph.nodes[target.childNodes[0].data]["pos"],
                    ),
                    3,
                )
            else:
                latlong1 = graph.nodes[source.childNodes[0].data]["pos"]
                latlong2 = graph.nodes[target.childNodes[0].data]["pos"]
                length = np.around(
                    math.sqrt(
                        (latlong1[0] - latlong2[0]) ** 2
                        + (latlong1[1] - latlong2[1]) ** 2
                    ),
                    3,
                )

            weight = 1.0
            graph.add_edge(
                source.childNodes[0].data,
                target.childNodes[0].data,
                id=link.getAttribute("id"),
                weight=weight,
                length=length,
                index=idx,
            )

    return graph


def read_txt_file(file):
    graph = nx.Graph()
    num_nodes = 0
    num_links = 0
    id_link = 0
    with open(file, "r") as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith("#")]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                for id in range(1, num_nodes + 1):
                    graph.add_node(str(id), name=str(id))
            elif idx == 1:
                num_links = int(line)
            elif len(line) > 1:
                info = line.replace("\n", "").split(" ")
                graph.add_edge(
                    info[0],
                    info[1],
                    id=id_link,
                    index=id_link,
                    weight=1,
                    length=int(info[2]),
                )
                id_link += 1

    return graph


def get_topology(file_name, topology_name, modulations, k_paths=5):
    k_shortest_paths = {}
    if file_name.endswith(".txt"):
        topology = read_txt_file(file_name)
    else:
        raise ValueError("Supplied topology is unknown")
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight="length")
                print(n1, n2, len(paths))
                lengths = [
                    get_path_weight(topology, path, weight="length") for path in paths
                ]
                if modulations is not None:
                    selected_modulations = [
                        get_best_modulation_format(length, modulations)
                        for length in lengths
                    ]
                else:
                    selected_modulations = [None for _ in lengths]
                objs = []

                for path, length, modulation in zip(
                    paths, lengths, selected_modulations
                ):
                    objs.append(
                        Path(
                            path_id=idp,
                            node_list=path,
                            hops=len(path) - 1,
                            length=length,
                            best_modulation=modulation,
                        )
                    )  
                    print("\t", objs[-1])
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    return topology


if __name__ == "__main__":
    # default values
    k_paths = 5
    topology_file = "nsfnet_chen.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--k_paths",
        type=int,
        default=k_paths,
        help="Number of k-shortest-paths to be considered (default={})".format(k_paths),
    )
    parser.add_argument(
        "-t",
        "--topology",
        default=topology_file,
        help="Network topology file to be used. Default: {}".format(topology_file),
    )

    args = parser.parse_args()

    topology_path = pathlib.Path(args.topology)

    topology = get_topology(
        args.topology, topology_path.stem.upper(), modulations, args.k_paths
    )

    file_name = topology_path.stem + "_" + str(k_paths) + "-paths"
    if modulations is not None:
        file_name += "_" + str(len(modulations)) + "-modulations"
    file_name += ".h5"

    output_file = topology_path.parent.resolve().joinpath(file_name)
    with open(output_file, "wb") as f:
        pickle.dump(topology, f)

    print("done for", topology)

