#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:14:45 2023

@author: ducoin
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import angular_separation
import copy


def cluster_data(tiles, threshold=5.0, start=0):
    """set positional data into clusters

    threshold : maximum angular separation between clusters points [°]

    Clusterization is made through a DBSCAN-like algorithm (see https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
    """
    ra_attr = "RA"
    dec_attr = "DEC"

    data = copy.deepcopy(tiles)
    data[ra_attr] = np.deg2rad(data[ra_attr].to_numpy())
    data[dec_attr] = np.deg2rad(data[dec_attr].to_numpy())

    max_dist = np.deg2rad(threshold)

    def get_cluster(data, max_dist):
        """Find the cluster associated to each data point"""
        total = data.shape[0]
        data["cluster"] = np.zeros(total)

        def iterate_cores(
            cores,
            data,
            max_dist,
            mask_no_cluster,
            mask_curr_cluster=np.array([]).astype(int),
        ):
            """Find the neighbour-points within [max_dist] angular distance for each [cores]-point of [data] not yet in another cluster [mask_no_cluster] nor in the current cluster [mask_curr_cluster],"""
            if mask_no_cluster.shape[0] <= 1:
                return mask_no_cluster

            # Iteration sur chaque core
            for ii in range(cores.shape[0]):
                core = cores.iloc[[ii]]
                ra, dec = core[ra_attr].iloc[0], core[dec_attr].iloc[0]
                mask_filtered = np.delete(
                    mask_no_cluster, np.isin(mask_no_cluster, mask_curr_cluster)
                )
                size = data.loc[mask_filtered, :].shape[0]
                dist = angular_separation(
                    data.loc[mask_filtered, :][ra_attr].to_numpy(),
                    data.loc[mask_filtered, :][dec_attr].to_numpy(),
                    ra * np.ones(size),
                    dec * np.ones(size),
                )

                # Ajoute au masque du cluster les bounders du point central
                (mask_bounders,) = np.where(dist < max_dist)
                mask_curr_cluster = np.concatenate(
                    (mask_curr_cluster, mask_filtered[mask_bounders])
                )
                bounders = data.loc[mask_filtered[mask_bounders], :]
                mask_curr_cluster = iterate_cores(
                    bounders,
                    data,
                    max_dist,
                    mask_no_cluster,
                    mask_curr_cluster=mask_curr_cluster,
                )

            return mask_curr_cluster

        num_cluster = 0
        while np.count_nonzero(data[["cluster"]].to_numpy()) != total:
            num_cluster += 1

            mask_no_cluster = np.where(data[["cluster"]].to_numpy() == 0)[0]
            core = data.loc[mask_no_cluster, :].iloc[[0]]
            mask_cluster = iterate_cores(core, data, max_dist, mask_no_cluster)
            data.loc[mask_cluster, "cluster"] = (
                np.ones(mask_cluster.shape[0]) * num_cluster
            )

        # Compter les clusters
        uniq_clusters = np.unique(data["cluster"].to_numpy())
        sumgal = 0

        for value in uniq_clusters:
            size = data[data["cluster"] == value].shape[0]
            sumgal += size
            print(f"Cluster n°{value} : {size} points [{sumgal}/{total}]")

    get_cluster(data, max_dist)
    tiles["cluster"] = data["cluster"]

    return tiles


def plot_slew(tiles, ax, until_tile=None, slew_constraint=5):
    """Plot angular distance between consecutive tiles
    Stem plot with a horizontal line for visualizing the slew constraint
    """
    if tiles[:until_tile].shape[0] <= 1:
        return

    # Compute angular distance
    dist = get_distance_between_tiles(tiles)[:until_tile]

    # Add hline
    if not ax.has_data():
        print_axhline = True
    else:
        print_axhline = False

    # Plot
    ax.stem(np.arange(1, 1 + dist.shape[0]), dist)

    if print_axhline:
        ax.axhline(y=slew_constraint, color="k", ls=":", lw=3)
        # ax.annotate(f"{slew_constraint}° constraint", color='k', xy=(0, slew_constraint), horizontalalignment='left', verticalalignment='bottom', xycoords='data', xytext=(2,2), textcoords='offset points', fontsize=18)


def get_distance_between_tiles(tiles):
    """compute the angular distance (in degree) between consecutive tiles"""

    ras = np.deg2rad(tiles["RA"].to_numpy())
    decs = np.deg2rad(tiles["DEC"].to_numpy())
    ang = angular_separation(ras[1:], decs[1:], ras[:-1], decs[:-1])

    return np.rad2deg(ang)


def plot_cluster(tiles, ax, until_tile=None, **kwargs):
    """Plot tiles and their associated cluster"""
    if tiles[:until_tile].shape[0] <= 1:
        return

    # Plot tiles
    ra, dec, prob, cluster = (
        tiles["RA"].to_numpy(),
        tiles["DEC"].to_numpy(),
        tiles["Prob"].to_numpy(),
        tiles["cluster"].to_numpy(),
    )

    ax.scatter(
        ra,
        dec,
        s=300 * prob + 50,
        c=cluster,
        cmap="gist_rainbow",
        alpha=0.7,
        edgecolors="k",
    )
    ax.axis("equal")


def set_ax_params(ax, xlabel, ylabel, legend=True, grid=True):
    """Set xlabel, ylabel, fontsize of labels & ticks. Optionnaly set the legend and the grid for any matplotlib.axes.Axes"""
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=20)

    if legend:
        ax.legend(fontsize=20)
    if grid:
        ax.grid()


def sequence_cluster(tiles, cluster, slew_constraint=5.0, tile_start=0):
    """sort each tile from a given [cluster] in order to respect the [slew_constraint] in degree while maximizing the probability observed
    Starting tile [tile_start] is by default 0, the best tile (with the best grade).
    It can be changed for optimization purposes.

    The next tile is determined with a criterion to minimize
    (for example, the distance to the cluster's barycenter, the distance to the current tile, the inverse of the grade, ...)
    """
    slew_constraint = np.deg2rad(slew_constraint)
    df = tiles

    # Get tiles of the given cluster
    (mask_cluster,) = np.where(df["cluster"].to_numpy() == cluster)
    df_clust = df.loc[mask_cluster, :]
    clust_size = df_clust.shape[0]
    tile = df_clust.iloc[[tile_start]]

    mask_tile_done = np.array([tile.index[0]])

    while mask_tile_done.shape[0] != clust_size:
        # ra, dec de la tuile actuelle
        ra, dec = tile["RA"].iloc[0], tile["DEC"].iloc[0]
        mask_cluster = np.delete(mask_cluster, np.isin(mask_cluster, mask_tile_done))

        size = mask_cluster.shape[0]
        dist = angular_separation(
            df_clust.loc[mask_cluster, :]["RA"].to_numpy(),
            df_clust.loc[mask_cluster, :]["DEC"].to_numpy(),
            ra * np.ones(size),
            dec * np.ones(size),
        )
        # Tuiles du voisinage, ie respectant la contrainte
        (mask_near_tile,) = np.where(dist < slew_constraint)

        if mask_near_tile.shape[0] == 0:
            # Aucune tuile au voisinage : on passe à la tuile suivante dans l'ordre (ie avec le grade le plus grand)
            tile = df_clust.loc[mask_cluster, :].iloc[[0]]

        elif mask_near_tile.shape[0] == 1:
            # Une unique tuile au voisinage : on passe à celle-ci
            tile = df_clust.loc[mask_cluster[mask_near_tile], :]

        else:
            # Plusieurs tuiles au voisinage :
            # on choisit la meilleure en fonction de son grade, sa distance au barycentre du cluster, sa distance à la tuile
            tiles = df_clust.loc[mask_cluster[mask_near_tile], :]
            ras, decs, probs = (
                tiles["RA"].to_numpy(),
                tiles["DEC"].to_numpy(),
                tiles["Prob"].to_numpy(),
            )

            barycenter_ra = np.average(ras, weights=probs)
            barycenter_dec = np.average(decs, weights=probs)
            dist_to_barycenter = angular_separation(
                ras,
                decs,
                barycenter_ra * np.ones(tiles.shape[0]),
                barycenter_dec * np.ones(tiles.shape[0]),
            )

            # Criterion to minimize : minimum dist_to_barycenter, maximum Smass
            criterion = dist_to_barycenter / probs
            # criterion = dist[mask_near_tile] * dist_to_barycenter / probs
            tile = tiles.iloc[[np.argsort(criterion)[0]]]

        mask_tile_done = np.concatenate((mask_tile_done, np.array([tile.index[0]])))

    return mask_tile_done


def sequence_all_clusters(
    tiles, slew_constraint=5.0, doOptimization=False, save=True, path_save="."
):
    """call sequence_cluster() for each cluster found with self.cluster_data()
    doOptimization = [bool] : run sequence_cluster either starting with the tile that has the maximum grade (False) or starting with all the tiles within each cluster (True)
    save = [bool]           : save the new sorted tiles.dat in another file, "tiles_cluster.dat"
    """
    df = tiles

    tiles["RA"] = np.deg2rad(tiles["RA"].to_numpy())
    tiles["DEC"] = np.deg2rad(tiles["DEC"].to_numpy())

    # Sort clusters by their total grade
    clusters = np.unique(df["cluster"].to_numpy())
    total_grade = np.zeros_like(clusters)

    for ii, cluster in enumerate(clusters):
        total_grade[ii] = np.sum(df[df["cluster"] == cluster]["Prob"].to_numpy())

    sort_mask = np.argsort(total_grade)[::-1]
    order_mask = np.array([])

    # Iterate through clusters from the best to the worst
    for cluster in clusters[sort_mask]:

        # Optimization : call sequence cluster with all the tiles of the cluster as the starting tile of the sequence
        if doOptimization:
            cluster_size = df[df["cluster"] == cluster].shape[0]
            tile_to_start = np.arange(cluster_size)

            min_nb_slew_exceed = cluster_size
            max_grade_integral = 0

            for ii, idx in enumerate(tile_to_start):
                mask_cluster_tile = sequence_cluster(
                    tiles, cluster, slew_constraint=slew_constraint, tile_start=idx
                )

                # Get each mask associated Smass per tile integral & the amount of time the slew constraint isn't respected
                ras = tiles.loc[mask_cluster_tile, :]["RA"].to_numpy()
                decs = tiles.loc[mask_cluster_tile, :]["DEC"].to_numpy()
                ang = np.rad2deg(
                    angular_separation(ras[1:], decs[1:], ras[:-1], decs[:-1])
                )

                (slew_exceed,) = np.where(ang > slew_constraint)
                nb_slew_exceed = slew_exceed.shape[0]
                grade_integral = np.sum(
                    tiles.loc[mask_cluster_tile, :]["Prob"].to_numpy().cumsum()
                )

                # Keep the sequence found if it minimizes the amount of slew exceeding the constraint and then maximizes the grade per tile integral
                if nb_slew_exceed < min_nb_slew_exceed or (
                    grade_integral > max_grade_integral
                    and nb_slew_exceed <= min_nb_slew_exceed
                ):
                    min_nb_slew_exceed = nb_slew_exceed
                    max_grade_integral = grade_integral
                    mask_cluster = mask_cluster_tile
                    print(
                        f"Cluster {cluster} best starting tile: idx n°{idx} with {nb_slew_exceed} slew errors & Smass integral of : {grade_integral:.6f}"
                    )

            # Best mask maximize the area under the Smass cumsum curve
            order_mask = np.concatenate((order_mask, mask_cluster))

        # No optimization : starting tile is the first tile by default
        else:
            mask_cluster = sequence_cluster(
                tiles, cluster, slew_constraint=slew_constraint
            )
            order_mask = np.concatenate((order_mask, mask_cluster))

    # Update tiles
    tiles = tiles.loc[order_mask, :]
    tiles["RA"] = np.rad2deg(tiles["RA"].to_numpy())
    tiles["DEC"] = np.rad2deg(tiles["DEC"].to_numpy())
    tiles["cluster"] = tiles["cluster"].astype(int)
    # Save the sorted tiles dataFrame
    if save:
        tiles.to_csv(path_save / Path("sequenced_tiles.dat"), sep=" ", index=False)


if __name__ == "__main__":
    ################################################################ MAIN

    # turn off interactive mode of matplotlib
    plt.ioff()

    # read the gwemopt tiles.dat output file
    path_tile = (
        "/home/ducoin/gwemopt-mxt.git/on_the_fly_run/test_cut_86400_MXT/tiles.dat"
    )
    path_save = "/home/ducoin/gwemopt-mxt.git/on_the_fly_run/test_cut_86400_MXT/display"

    slew_constraint = 5.0  # deg

    tiles = pd.read_csv(path_tile, delimiter=" ")

    # add a columns with the cumulative sum of proba
    tiles["prob_sum"] = tiles["prob"].to_numpy().cumsum()

    #######plot the tiles distance with the raw output of gwemopt

    fig, ax = plt.subplots(figsize=(15, 10))
    # the slew constraint is display as a dashed line, change the value according to your constraint
    # until_tile stop the plot at a given number of tile if required
    plot_slew(tiles, ax, until_tile=None, slew_constraint=slew_constraint)
    set_ax_params(
        ax,
        "Tile index",
        "Angular distance between consecutive tiles [deg]",
        legend=False,
    )
    plt.savefig(path_save + "/raw_slew")

    #######identify cluster of pointing

    # add a column in the tiles dataframe with the cluster ID
    tiles = cluster_data(tiles, threshold=5.0, start=0)
    # plot the identify clusters
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_cluster(tiles, ax, until_tile=None)
    set_ax_params(ax, "right ascension [deg]", "declination [deg]", legend=False)
    ax.set_ylim(-90, 90)
    plt.savefig(path_save + "/clusters.png")

    #######optimise the slew constraint cluster by cluster and save the new observation in sequenced_tiles.dat
    sequence_all_clusters(
        tiles,
        slew_constraint=slew_constraint,
        doOptimization=False,
        save=True,
        path_save=path_save,
    )

    #######plot the tiles distance with the new sequenced observations

    sequenced_tiles = pd.read_csv(path_save + "/sequenced_tiles.dat", delimiter=" ")
    fig, ax = plt.subplots(figsize=(15, 10))
    # the slew constraint is display as a dashed line, change the value according to your constraint
    # until_tile stop the plot at a given number of tile if required
    plot_slew(sequenced_tiles, ax, until_tile=None, slew_constraint=slew_constraint)
    set_ax_params(
        ax,
        "Tile index",
        "Angular distance between consecutive tiles [deg]",
        legend=False,
    )
    plt.savefig(path_save + "/sequenced_slew")
