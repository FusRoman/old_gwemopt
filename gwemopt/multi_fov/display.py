#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
import pandas as pd
import seaborn as sns

from astropy.coordinates import angular_separation


def search_tree(path):
    """Search output files within directory path and extract data
    Catalog of galaxies : *.catalog.csv
    Gwemopt tiles       : *.tiles.dat

    Returns pandas.dataFrame with 5 columns:
    - group      : usually the tilesType provided (galaxy, moc, etc.)
    - skymap     : skymap of the given simulation
    - telescope  : telescope used (SVOM-MXT, SVOM-MXT-VT_sun, etc.)
    - gps        : [int] GPS time
    - simulation : Simulation object
    """
    path_catalogs = [
        file
        for path, subdir, files in os.walk(path)
        for file in glob(os.path.join(path, "catalog.csv"))
    ]

    path_tiles = [
        file
        for path, subdir, files in os.walk(path)
        for file in glob(os.path.join(path, "tiles.dat"))
    ]

    if len(path_catalogs) != len(path_tiles):
        print(
            f"Error in output files : the number of catalog files ({len(path_catalogs)}) and tiles file ({len(path_tiles)})are not equal !"
        )
        exit(0)

    data = []
    for ii in range(len(path_catalogs)):
        simulation = Simulation(path_catalogs[ii], path_tiles[ii])
        data.append(
            [
                simulation.group,
                simulation.skymap,
                simulation.telescope,
                simulation.gps,
                simulation,
            ]
        )

    dataFrame = pd.DataFrame(
        data, columns=["group", "skymap", "telescope", "gps", "simulation"]
    )

    return dataFrame


def set_ax_params(ax, xlabel, ylabel, legend=True, grid=True):
    """Set xlabel, ylabel, fontsize of labels & ticks. Optionnaly set the legend and the grid for any matplotlib.axes.Axes"""
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=20)

    if legend:
        ax.legend(fontsize=20)
    if grid:
        ax.grid()


def save(figure, name, simulation=None, group=None, save_directory="fig"):
    """Save [figure] in the [save_directory] folder (or create if if it doesn't exist) under the right [name].png"""
    if simulation:
        path = f"{save_directory}/{simulation.skymap}/{name}_{simulation.skymap}_{simulation.group}"
    elif group:
        path = f"{save_directory}/{name}_{group}"
    else:
        return

    # MKDIR
    for directory in [
        "/".join(path.split("/")[: i + 1]) for i in range(len(path.split("/")[:-1]))
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save figure
    figure.savefig(path)

    plt.close(figure)


class Simulation:
    FoV_MXT = 1.0667
    FoV_VT = 0.4333

    def __init__(self, path_catalog, path_tile):
        # Simulation parameters
        subdirs = path_catalog.split("/")
        self.gps = subdirs[-2]
        self.telescope = subdirs[-3]
        self.skymap = subdirs[-4]
        self.group = subdirs[-5]
        self.output = "/".join(subdirs[:-1])

        # Set the fields of view of the telescope used
        self.FoV = None
        self.FoV_center = None
        self.get_telescope_config()

        # Catalog file ==> pandas.dataFrame
        self.catalog = pd.read_csv(path_catalog)
        self.catalog.columns = self.catalog.columns.str.replace(" ", "")

        # Tiles file ==> pandas.
        # If the tiles have already been sorted with the cluster method, the sorted file is used instead (ie "tiles_cluster.dat")
        if "tiles_cluster.dat" in os.listdir(self.output):
            print("Tiles clustered file found")
            path_tile = self.output + "/tiles_cluster.dat"

        # If the group is "moc" there is no "galaxies" nor "ngal" columns in the "tiles.dat" file thus it must be added
        if "moc" in self.group:
            self.tiles = pd.read_csv(path_tile, delimiter=" ")
            self.tiles["galaxies"] = self.extract_galaxies(self.FoV)
            self.tiles["ngal"] = [len(el) for el in self.tiles["galaxies"].to_list()]
        else:
            self.tiles = pd.read_csv(
                path_tile,
                delimiter=" ",
                converters={"galaxies": lambda x: [int(el) for el in x.split(",")]},
            )
        self.tiles["prob_sum"] = self.tiles["prob"].to_numpy().cumsum()
        self.tiles["ngal_sum"] = self.tiles["ngal"].to_numpy().cumsum()
        self.ntiles = self.tiles.shape[0]

    @staticmethod
    def heatmap_gps(
        ax, simus, simu_ref, tiles=None, attr="prob_sum", norm=True, iso=[]
    ):
        """Seaborn heatmap to plot the effect of the sun constraint (ie multiple Simulations with different GPS time against a single Simulation reference with no sun constraint)
        Each column of the heatmap represents a number of consecutive [tiles]
        [iso] is a list of floats to add a contour plot on the heatmap to enhance the visualization of a threshold
        """
        # Set arrays
        if tiles is None:
            tiles = np.arange(simu_ref.ntiles)
        length = len(simus)
        size = tiles.shape[0]

        array = np.zeros((length, size))
        array_ref = simu_ref.tiles[attr].to_numpy()[
            np.where(tiles < simu_ref.ntiles, tiles, -1)
        ]
        for ii in range(length):
            simu_attr = simus[ii].tiles[attr].to_numpy()
            if simu_attr.shape[0]:
                array[ii, :] = simu_attr[np.where(tiles < simus[ii].ntiles, tiles, -1)]
        data = array - array_ref

        # Normalize data
        if norm:
            data /= np.maximum(array, array_ref)

        df = pd.DataFrame(data, index=[simu.gps for simu in simus])
        df.index = df.index.astype(int)
        df = df.sort_index()
        df = df.set_axis(np.linspace(0, 12, df.shape[0]), axis=0)

        # Heatmap
        ax_heatmap = sns.heatmap(
            df,
            annot=False,
            cmap="bwr_r",
            cbar_kws={"label": "Grade difference"},
            center=-0.5,
            ax=ax,
            vmin=-1,
            vmax=0,
            xticklabels=int(df.shape[1] / 10),
        )
        ax_colorbar = ax_heatmap.figure.axes[1]
        ax_colorbar.yaxis.label.set_size(22)
        ax_colorbar.tick_params(labelsize=20)

        def define_y_axis(ax, ticks, label):
            # Change y-axis with custom [ticks] and [label]
            ax.set_yticks([])
            new_ax = ax.twinx()
            new_ax.set_yticks(ticks)
            new_ax.set_yticklabels([str(i) for i in ticks[::-1]], fontsize=22)
            new_ax.yaxis.set_label_position("left")
            new_ax.yaxis.tick_left()
            new_ax.set_ylabel(label, fontsize=22)

        # Replace GPS time of the y-axis with Months (from 0 to 12)
        # /!\ Need to ensure the GPS list has a time span of 1 year
        define_y_axis(ax, np.arange(13), "Months")
        ax.set_xlabel("Number of tiles", fontsize=22)
        ax.tick_params(labelsize=20)

        # Add iso-contour lines
        def add_iso(ax, colorbar, df, value, ls="solid", lw=2):
            # Add contour line as a threshold for a given iso value
            # Find complementary color for [value] on the cmap used
            cmap = colormaps["bwr_r"]
            color = np.array(cmap(1 + value))
            c_color = 1 - color
            c_color[3] = 1  # Transparency = 1

            z = df.to_numpy()
            ax.contour(
                np.linspace(0, size, size),
                np.linspace(0, length, length),
                z,
                levels=[value],
                colors=[c_color],
                linestyles=ls,
                linewidths=lw,
            )

            colorbar.plot([0, 1], [value] * 2, c=c_color)

        for value in iso:
            add_iso(ax, ax_colorbar, df, -value)

        # Add a vertical threshold at 70 tiles
        ax.axvline(x=70, color="k", ls=":", lw=2)
        ax.annotate(
            "70 tiles",
            xy=(70, 0),
            color="k",
            horizontalalignment="center",
            verticalalignment="bottom",
            xycoords="data",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=18,
        )

    @staticmethod
    def heatmap_skymaps(ax, simus, simus_ref, tiles=None, attr="prob_sum", norm=True):
        """Seaborn heatmap to plot the relative difference of a Simulation against a Simulation reference.
        Each column of the heatmap represents a number of consecutive [tiles]
        """
        # Check if simus and simus_ref have same length
        if len(simus) != len(simus_ref):
            print(f"Can't compute heatmap with {len(simus)} and {len(simus_ref)}")
            return

        # Set arrays
        if tiles is None:
            tiles = np.arange(max(simus.ntiles, simus_ref.ntiles))
        length = len(simus)
        size = tiles.shape[0]

        # To avoid cases where there is no tiles observed either in simus or simus_ref (ie in both cases there isn't any observable galaxy for GWEMOPT) :
        # Arrays are set by default to -1 instead of 0, which avoids a division by 0 when normalizing data
        array = np.ones((length, size)) * -1
        array_ref = np.ones((length, size)) * -1
        for ii in range(length):
            if simus[ii].ntiles:
                array[ii, :] = (
                    simus[ii]
                    .tiles[attr]
                    .to_numpy()[np.where(tiles < simus[ii].ntiles, tiles, -1)]
                )
            if simus[ii].ntiles:
                array_ref[ii, :] = (
                    simus_ref[ii]
                    .tiles[attr]
                    .to_numpy()[np.where(tiles < simus_ref[ii].ntiles, tiles, -1)]
                )
        data = array - array_ref

        # Normalize data
        if norm:
            data /= np.maximum(array, array_ref)

        df = pd.DataFrame(
            data,
            columns=[f"{str(el)} tiles" if el >= 0 else f"max tiles" for el in tiles],
            index=[simu.skymap for simu in simus],
        ).sort_index()

        # Heatmap
        ax_heatmap = sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="bwr_r",
            center=0,
            linewidths=1,
            linecolor="k",
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax_colorbar = ax_heatmap.figure.axes[1]

    def get_telescope_config(self):
        """Search for telescope FoVs and FoVs_edge within telescope config files"""
        path = f"config/{self.telescope}.config"
        df = pd.read_csv(path, delimiter=" ", index_col=0, header=None).T
        if "FOV" in df.columns:
            self.FoV = float(df["FOV"].iloc[0])
        else:
            self.FoV = self.FoV_MXT
        if "FOV_edge" in df.columns:
            self.FoV *= float(df["FOV_edge"].iloc[0])

        if "FOV_center" in df.columns:
            self.FoV_center = float(df["FOV_center"].iloc[0])
        else:
            self.FoV_center = self.FoV_VT
        if "FOV_center_edge" in df.columns:
            self.FoV_center *= float(df["FOV_center_edge"].iloc[0])

    def get_distance_between_tiles(self):
        """compute the angular distance (in degree) between consecutive tiles"""
        ras = np.deg2rad(self.tiles["ra"].to_numpy())
        decs = np.deg2rad(self.tiles["dec"].to_numpy())
        ang = angular_separation(ras[1:], decs[1:], ras[:-1], decs[:-1])
        return np.rad2deg(ang)

    def extract_galaxies(self, FoV, doOverlap=False, until_tile=None):
        """extract galaxies from each tile that are within FoV square range.
        doOverlap = [bool]         : get overlapping galaxies on each tile it appears.
        until_tile = [int or None] : use only the first N tiles

        Returns a [list of int] for each galaxy id found

        /!\ This method doesn't work properly for tiles that have a corner with dec > 87°
        To avoid those cases, one should implement the same method that is used in GWEMOPT tiles.py
        """
        tiles = self.tiles[:until_tile]
        FoV /= 2.0
        galaxies = []

        # FoV center and limits of the square
        ra_center = tiles["ra"].to_numpy()
        dec_center = tiles["dec"].to_numpy()

        ra_limits = (
            ra_center - FoV / np.cos(np.deg2rad(dec_center)),
            ra_center + FoV / np.cos(np.deg2rad(dec_center)),
        )
        dec_limits = (dec_center - FoV, dec_center + FoV)

        # Extract galaxies from catalog.csv
        ra_galaxies = self.catalog["RAJ2000"].to_numpy()
        dec_galaxies = self.catalog["DEJ2000"].to_numpy()
        id_galaxies = self.catalog["id"].to_numpy()

        for ii in range(tiles.shape[0]):
            idx_ra = np.where(
                (ra_galaxies >= ra_limits[0][ii]) & (ra_galaxies <= ra_limits[1][ii])
            )
            idx_dec = np.where(
                (dec_galaxies >= dec_limits[0][ii])
                & (dec_galaxies <= dec_limits[1][ii])
            )
            mask = np.intersect1d(idx_ra, idx_dec)
            galaxies.append(id_galaxies[mask].tolist())

        # Overlapping between tiles
        if not doOverlap:
            unique_galaxies = []
            for ii in range(tiles.shape[0]):
                galaxies[ii] = [
                    galaxy for galaxy in galaxies[ii] if galaxy not in unique_galaxies
                ]
                unique_galaxies.extend(galaxies[ii])

        return galaxies

    def search_catalog(self, galaxies, *attribute):
        """search galaxies within catalog.csv and return the associated [attribute]s if they exist
        galaxies   : list of int
        *attribute : strings corresponding to a catalog.csv column

        Returns a numpy.array() if there is a single argument for [attribute], or a tuple of numpy.array() if multiples are provided
        """
        if not galaxies:
            attribute_galaxies = (np.array([0]) for attr in attribute)
        else:
            attribute_galaxies = (
                self.catalog[attr].to_numpy()[np.array(galaxies) - 1]
                for attr in attribute
            )  # -1 because galaxies id between catalog.csv and tiles.dat are shifted by 1

        if len(attribute) == 1:
            return list(attribute_galaxies)[0]
        else:
            return attribute_galaxies

    def cluster_data(self, datatype="tile", threshold=5.0, start=0):
        """set positional data into clusters
        datatype :  "galaxy" to get galaxies' clusters
                    "tile"   to get tiles' clusters
        threshold : maximum angular separation between clusters points [°]

        Clusterization is made through a DBSCAN-like algorithm (see https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
        """
        if datatype == "tile":
            ra_attr = "ra"
            dec_attr = "dec"
        else:
            ra_attr = "RAJ2000"
            dec_attr = "DEJ2000"

        data = self.tiles
        data[ra_attr] = np.deg2rad(data[ra_attr].to_numpy())
        data[dec_attr] = np.deg2rad(data[dec_attr].to_numpy())

        max_dist = np.deg2rad(threshold)

        # Find clusters
        print(f"\n{self.skymap :}")
        print(f"{data.shape[0]} galaxies au total")

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
                print(f"Cluster n°{value} : {size} galaxies [{sumgal}/{total}]")

        get_cluster(data, max_dist)

    def sequence_cluster(self, cluster, slew_constraint=5.0, tile_start=0):
        """sort each tile from a given [cluster] in order to respect the [slew_constraint] in degree while maximizing the probability observed
        Starting tile [tile_start] is by default 0, the best tile (with the best grade).
        It can be changed for optimization purposes.

        The next tile is determined with a criterion to minimize
        (for example, the distance to the cluster's barycenter, the distance to the current tile, the inverse of the grade, ...)
        """
        slew_constraint = np.deg2rad(slew_constraint)
        df = self.tiles

        # Get tiles of the given cluster
        (mask_cluster,) = np.where(df["cluster"].to_numpy() == cluster)
        df_clust = df.loc[mask_cluster, :]
        clust_size = df_clust.shape[0]
        tile = df_clust.iloc[[tile_start]]

        mask_tile_done = np.array([tile.index[0]])

        while mask_tile_done.shape[0] != clust_size:
            # ra, dec de la tuile actuelle
            ra, dec = tile["ra"].iloc[0], tile["dec"].iloc[0]
            mask_cluster = np.delete(
                mask_cluster, np.isin(mask_cluster, mask_tile_done)
            )

            size = mask_cluster.shape[0]
            dist = angular_separation(
                df_clust.loc[mask_cluster, :]["ra"].to_numpy(),
                df_clust.loc[mask_cluster, :]["dec"].to_numpy(),
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
                    tiles["ra"].to_numpy(),
                    tiles["dec"].to_numpy(),
                    tiles["prob"].to_numpy(),
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
        self, slew_constraint=5.0, doOptimization=False, save=True
    ):
        """call self.sequence_cluster() for each cluster found with self.cluster_data()
        doOptimization = [bool] : run sequence_cluster either starting with the tile that has the maximum grade (False) or starting with all the tiles within each cluster (True)
        save = [bool]           : save the new sorted tiles.dat in another file, "tiles_cluster.dat"
        """
        # Define clusters for the tiles
        self.cluster_data(threshold=slew_constraint)
        df = self.tiles

        # Sort clusters by their total grade
        clusters = np.unique(df["cluster"].to_numpy())
        total_grade = np.zeros_like(clusters)

        for ii, cluster in enumerate(clusters):
            total_grade[ii] = np.sum(df[df["cluster"] == cluster]["prob"].to_numpy())

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
                    mask_cluster_tile = self.sequence_cluster(
                        cluster, slew_constraint=slew_constraint, tile_start=idx
                    )

                    # Get each mask associated Smass per tile integral & the amount of time the slew constraint isn't respected
                    ras = self.tiles.loc[mask_cluster_tile, :]["ra"].to_numpy()
                    decs = self.tiles.loc[mask_cluster_tile, :]["dec"].to_numpy()
                    ang = np.rad2deg(
                        angular_separation(ras[1:], decs[1:], ras[:-1], decs[:-1])
                    )

                    (slew_exceed,) = np.where(ang > slew_constraint)
                    nb_slew_exceed = slew_exceed.shape[0]
                    grade_integral = np.sum(
                        self.tiles.loc[mask_cluster_tile, :]["prob"].to_numpy().cumsum()
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
                mask_cluster = self.sequence_cluster(
                    cluster, slew_constraint=slew_constraint
                )
                order_mask = np.concatenate((order_mask, mask_cluster))

        # Update self.tiles
        self.tiles = self.tiles.loc[order_mask, :]
        self.tiles["ra"] = np.rad2deg(self.tiles["ra"].to_numpy())
        self.tiles["dec"] = np.rad2deg(self.tiles["dec"].to_numpy())
        self.tiles["cluster"] = self.tiles["cluster"].astype(int)

        # Save the sorted tiles dataFrame
        if save:
            with open(f"{self.output}/tiles_cluster.dat", "w") as filename:
                ids = self.tiles["id"].to_numpy()
                ra = self.tiles["ra"].to_numpy()
                dec = self.tiles["dec"].to_numpy()
                prob = self.tiles["prob"].to_numpy()
                clus = self.tiles["cluster"].to_numpy()
                ngal = self.tiles["ngal"].to_numpy()
                gals = self.tiles["galaxies"].to_numpy()

                filename.write(f"id ra dec prob cluster ngal galaxies\n")
                for ii in range(self.tiles.shape[0]):
                    filename.write(
                        f"{ids[ii]} {ra[ii]:.5f} {dec[ii]:.5f} {prob[ii]:.5f} {clus[ii]} {ngal[ii]} {','.join([str(gal) for gal in gals[ii]])}\n"
                    )

    def plot_grade(self, ax, FoV=None, until_tile=None, **kwargs):
        """Plot grade cumsum of the tiles, searching the galaxies that are in the [FoV]"""
        if not FoV:
            FoV = self.FoV

        # Galaxies for a given FoV
        galaxies = self.extract_galaxies(FoV, until_tile=until_tile)
        gal_tile = self.tiles["galaxies"].to_list()

        # Add hline y = 1
        if not ax.has_data():
            ax.axhline(y=1, color="k", ls="-.", lw=2)

        # Compute Smass per tile
        size = len(galaxies)
        grade = np.zeros(size)
        for ii in range(size):
            grade[ii] = np.sum(self.search_catalog(galaxies[ii], "Smass"))

        # Plot
        ax.plot(np.arange(1, 1 + size), grade.cumsum(), **kwargs)

    def plot_ngal(self, ax, FoV=None, until_tile=None, **kwargs):
        """Plot number of galaxies cumsum of the tiles, searching the galaxies that are in the [FoV]"""
        if not FoV:
            FoV = self.FoV

        # Galaxies for a given FoV
        galaxies = self.extract_galaxies(FoV, until_tile=until_tile)

        # Add hline y = total number of galaxies within catalog.csv
        if not ax.has_data():
            ax.axhline(y=self.catalog.shape[0], color="k", ls="-.", lw=2)
            ax.annotate(
                f"{self.catalog.shape[0]} galaxies compatible in Mangrove",
                color="k",
                xy=(0, self.catalog.shape[0] / ax.get_ylim()[1]),
                horizontalalignment="left",
                verticalalignment="top",
                xycoords=ax.get_xaxis_transform(),
                xytext=(5, -5),
                textcoords="offset points",
                fontsize=16,
            )

        # Compute # of galaxies per tile
        size = len(galaxies)
        ngal = np.zeros(size)
        for ii in range(size):
            ngal[ii] = len(galaxies[ii])

        # Plot
        ax.plot(np.arange(1, 1 + size), ngal.cumsum(), **kwargs)

    def plot_FoV(self, ax, FoV=None, until_tile=None, norm=True):
        """Plot galaxies position (ra, dec) relatively to the tiles' center, for galaxies that are in the [FoV]
        norm = [bool] : normalize grade of the galaxies
        """
        if not FoV:
            FoV = self.FoV

        # Galaxies for a given FoV
        tiles = self.tiles[:until_tile]
        galaxies = self.extract_galaxies(FoV, doOverlap=True, until_tile=until_tile)
        galaxies_flat = [gal for per_tile in galaxies for gal in per_tile]

        # Compute radec of each galaxy
        size = len(galaxies_flat)
        if not size:
            return
        ra, dec = self.search_catalog(galaxies_flat, "RAJ2000", "DEJ2000")

        # Center of each galaxy's tile
        mask_tiles = np.array(
            [galaxies.index(per_tile) for per_tile in galaxies for gal in per_tile]
        )
        ra_center = tiles["ra"].to_numpy()[mask_tiles]
        dec_center = tiles["dec"].to_numpy()[mask_tiles]

        # Relative radec
        ra_rel = (ra - ra_center) * np.cos(np.deg2rad(dec_center))
        dec_rel = dec - dec_center

        # For non-unique galaxies (ie those seen in multiple FoVs), find out the closest to the center
        galaxies_flat = np.array(galaxies_flat)
        uniq, uniq_counts = np.unique(galaxies_flat, return_counts=True)
        mask_non_unique = np.isin(galaxies_flat, uniq[np.where(uniq_counts > 1)])

        # Infinity norm to get distance (ie max(ra,dec))
        # Infinity norm is used because the FoV is a square
        size_non_unique = np.unique(galaxies_flat[mask_non_unique]).shape[0]
        ra_rel_non_unique = np.zeros(size_non_unique)
        dec_rel_non_unique = np.zeros(size_non_unique)

        for ii, value in enumerate(np.unique(galaxies_flat[mask_non_unique])):
            ra_rel_gal = ra_rel[np.where(galaxies_flat == value)]
            dec_rel_gal = dec_rel[np.where(galaxies_flat == value)]
            dist = np.maximum(np.abs(ra_rel_gal), np.abs(dec_rel_gal))
            idx = np.argmin(dist)
            ra_rel_non_unique[ii] = ra_rel_gal[idx]
            dec_rel_non_unique[ii] = dec_rel_gal[idx]

        ra_rel = np.concatenate((ra_rel[~mask_non_unique], ra_rel_non_unique))
        dec_rel = np.concatenate((dec_rel[~mask_non_unique], dec_rel_non_unique))
        galaxies = np.concatenate(
            (galaxies_flat[~mask_non_unique], np.unique(galaxies_flat[mask_non_unique]))
        )
        prob = self.search_catalog(galaxies.tolist(), "Smass")

        # Add edges to delimit FoVs
        if not ax.has_data():
            ra_limits = np.array([-0.5, -0.5, 0.5, 0.5])
            dec_limits = np.array([-0.5, 0.5, 0.5, -0.5])

            ax.fill(
                ra_limits * Simulation.FoV_MXT,
                dec_limits * Simulation.FoV_MXT,
                facecolor="none",
                edgecolor="red",
                linewidth=1,
                label="MXT",
            )
            ax.fill(
                ra_limits * Simulation.FoV_VT,
                dec_limits * Simulation.FoV_VT,
                facecolor="none",
                edgecolor="blue",
                linewidth=1,
                label="VT",
            )
            ax.axis("equal")

        # Normalize grade
        if norm:
            prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))

        # Scatter plot with a different marker for the galaxies seen by MXT only and those seen by MXT & VT
        in_VT = np.where(
            np.maximum(np.abs(ra_rel), np.abs(dec_rel)) < self.FoV_center / 2.0,
            True,
            False,
        )
        ax.scatter(
            ra_rel[~in_VT],
            dec_rel[~in_VT],
            marker="o",
            s=prob[~in_VT] * 250 + 50,
            c=prob[~in_VT],
            cmap="Reds",
            alpha=0.7,
            edgecolors="k",
        )
        ax.scatter(
            ra_rel[in_VT],
            dec_rel[in_VT],
            marker="^",
            s=prob[in_VT] * 250 + 50,
            c=prob[in_VT],
            cmap="Reds",
            alpha=0.7,
            edgecolors="k",
        )

        # Scatter plot with the same marker
        # ax.scatter(ra_rel, dec_rel, marker="o", s=prob*250+50, c=prob, cmap='Reds', alpha=0.7, edgecolors='k')

    def plot_slew(self, ax, until_tile=None, slew_constraint=5, **kwargs):
        """Plot angular distance between consecutive tiles
        Stem plot with a horizontal line for visualizing the slew constraint
        """
        if self.tiles[:until_tile].shape[0] <= 1:
            return

        # Compute angular distance
        dist = self.get_distance_between_tiles()[:until_tile]

        # Add hline
        if not ax.has_data():
            print_axhline = True
        else:
            print_axhline = False

        # Plot
        ax.stem(np.arange(1, 1 + dist.shape[0]), dist, **kwargs)

        if print_axhline:
            ax.axhline(y=slew_constraint, color="k", ls=":", lw=3)
            # ax.annotate(f"{slew_constraint}° constraint", color='k', xy=(0, slew_constraint), horizontalalignment='left', verticalalignment='bottom', xycoords='data', xytext=(2,2), textcoords='offset points', fontsize=18)

    def plot_path(self, ax, until_tile=None, **kwargs):
        """Plot path of the telescope from first to last tile seen within each cluster"""
        if self.tiles[:until_tile].shape[0] <= 1:
            return

        # Compute angular distance
        dist = self.get_distance_between_tiles()[:until_tile]

        # Plot tiles
        ra, dec, prob = (
            self.tiles["ra"].to_numpy(),
            self.tiles["dec"].to_numpy(),
            self.tiles["prob"].to_numpy(),
        )
        ax.scatter(ra, dec, s=300 * prob + 80, c="white", alpha=0.7, edgecolors="k")
        ax.axis("equal")

        # Plot path within each clusters
        if "cluster" in self.tiles.columns:
            for cluster in np.unique(self.tiles["cluster"].to_numpy()):
                df_clust = self.tiles[self.tiles["cluster"] == cluster]
                ax.plot(
                    df_clust["ra"].to_numpy(),
                    df_clust["dec"].to_numpy(),
                    lw=1,
                    label=f"Cluster n°{cluster}",
                )

        else:
            ax.plot(self.tiles["ra"].to_numpy(), self.tiles["dec"].to_numpy(), lw=1)

    def plot_cluster(self, ax, until_tile=None, **kwargs):
        """Plot tiles and their associated cluster"""
        if self.tiles[:until_tile].shape[0] <= 1:
            return

        # Plot tiles
        ra, dec, prob, cluster = (
            self.tiles["ra"].to_numpy(),
            self.tiles["dec"].to_numpy(),
            self.tiles["prob"].to_numpy(),
            self.tiles["cluster"].to_numpy(),
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


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Display plots for GWEMOPT outputs. By default, it does grade, ngal, FoV and slew plots for each output found.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "output",
        default=None,
        help="set the path directory of gwemopt simulation output",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        default=False,
        help="do grade plots for each output",
    )
    parser.add_argument(
        "--ngal",
        action="store_true",
        default=False,
        help="do # of galaxy plots for each output",
    )
    parser.add_argument(
        "--fov", action="store_true", default=False, help="do FoV plots for each output"
    )
    parser.add_argument(
        "--slew",
        action="store_true",
        default=False,
        help="do slew plots for each output",
    )
    parser.add_argument(
        "--path",
        action="store_true",
        default=False,
        help="do path plots to show the sequence of targeting",
    )
    parser.add_argument(
        "--heatmap",
        dest="heatmap",
        metavar="reference_group",
        default=False,
        help="plot heatmap to compare each skymap against a reference",
    )
    parser.add_argument(
        "--max", dest="max_tiles", default=None, help="do plots for the first N tiles"
    )
    parser.add_argument(
        "--figdir",
        dest="figdir",
        default="fig",
        help="set directory path to save figures in",
    )
    parser.add_argument(
        "--sun", action="store_true", default=False, help="plot sun constraint heatmaps"
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="sort the tiles to optimize the slew constraint",
    )
    parser.add_argument(
        "--skymap",
        dest="skymap",
        default=None,
        help="restrict plots to the skymaps provided",
    )

    args = parser.parse_args()

    max_tiles = int(args.max_tiles) if args.max_tiles is not None else None

    # Find every simulation output
    df = search_tree(args.output)

    # Restrict dataFrame to given skymaps
    if args.skymap:
        maps_list = args.skymap.split(",")
        df = df[df["skymap"].isin(maps_list)]
        print(f"Skymaps: {np.unique(df['skymap'].to_numpy())}")

    # Do Plots
    print("\n== plots")
    simus_moc = df[df["group"] == "moc"]["simulation"].to_list()
    simus_GT = df[df["group"] != "moc"]["simulation"].to_list()
    simus = df["simulation"].to_list()

    # Minimize slew constraint
    if args.cluster:
        for simu in simus:
            # simu.sequence_tile()
            simu.sequence_all_clusters(doOptimization=False)
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_cluster(ax)
            set_ax_params(
                ax, "right ascension [deg]", "declination [deg]", legend=False
            )
            ax.set_ylim(-90, 90)
            save(fig, "cluster", simulation=simu, save_directory=args.figdir)

    # If there are both "galaxy" and "moc" within the output folder, plot each one on the same figure
    for simu_GT, simu_moc in zip(simus_GT, simus_moc):
        if not args.grade and not args.ngal:
            continue
        print(
            f"== Plots GT-moc : {simu_GT.skymap} [{simu_moc.skymap}] - {simu_GT.telescope} [{simu_moc.telescope}]"
        )

        # Grade
        if args.grade:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu_GT.plot_grade(
                ax, FoV=simu_GT.FoV, until_tile=max_tiles, label="Galaxy targeting"
            )
            simu_moc.plot_grade(
                ax, FoV=simu_moc.FoV, until_tile=max_tiles, label="Tiling"
            )

            # simu.plot_grade(ax, FoV=simu.FoV_center, until_tile=max_tiles, label='VT')
            set_ax_params(ax, "Number of tiles", "Cumulative grade observed")
            save(fig, "grade", simulation=simu_GT, save_directory=args.figdir)

        # Ngal
        if args.ngal:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu_GT.plot_ngal(
                ax, FoV=simu_GT.FoV, until_tile=max_tiles, label="Galaxy targeting"
            )
            simu_moc.plot_ngal(
                ax, FoV=simu_moc.FoV, until_tile=max_tiles, label="Tiling"
            )

            # simu.plot_ngal(ax, FoV=simu.FoV_center, until_tile=max_tiles, label='VT')
            set_ax_params(
                ax, "Number of tiles", "Cumulative number of galaxies observed"
            )
            save(fig, "ngal", simulation=simu_GT, save_directory=args.figdir)

    # For each simulation, do all plots
    for simu in simus:
        print(f"== Plots : {simu.group} - {simu.skymap} - {simu.telescope}")

        # Grade
        if args.grade:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_grade(ax, FoV=simu.FoV, until_tile=max_tiles, label="MXT")
            simu.plot_grade(ax, FoV=simu.FoV_center, until_tile=max_tiles, label="VT")
            set_ax_params(ax, "Number of tiles", "Cumulative grade observed")
            save(fig, "grade", simulation=simu, save_directory=args.figdir)

        # Ngal
        if args.ngal:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_ngal(ax, FoV=simu.FoV, until_tile=max_tiles, label="MXT")
            simu.plot_ngal(ax, FoV=simu.FoV_center, until_tile=max_tiles, label="VT")
            set_ax_params(
                ax, "Number of tiles", "Cumulative number of galaxies observed"
            )
            save(fig, "ngal", simulation=simu, save_directory=args.figdir)

        # FoV
        if args.fov:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_FoV(ax, FoV=simu.FoV, until_tile=max_tiles, norm=True)
            set_ax_params(ax, "right ascension [deg]", "declination [deg]")
            save(fig, "FoV", simulation=simu, save_directory=args.figdir)

        # Slew
        if args.slew:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_slew(ax, until_tile=max_tiles)
            set_ax_params(
                ax,
                "Tile index",
                "Angular distance between consecutive tiles [deg]",
                legend=False,
            )
            save(fig, "slew", simulation=simu, save_directory=args.figdir)

        # Path
        if args.path:
            fig, ax = plt.subplots(figsize=(15, 10))
            simu.plot_path(ax, until_tile=max_tiles)
            set_ax_params(ax, "right ascension [deg]", "declination [deg]")
            save(fig, "path", simulation=simu, save_directory=args.figdir)

    # Skymap heatmap against a group reference
    if args.heatmap:
        groups = np.unique(df["group"].to_numpy())
        for group in groups:
            if group == args.heatmap:
                continue
            # Smass
            fig, ax = plt.subplots(figsize=(15, 10))
            Simulation.heatmap_skymaps(
                ax,
                df[df["group"] == group]["simulation"].to_list(),
                df[df["group"] == args.heatmap]["simulation"].to_list(),
                tiles=np.array([5, 10, 70, -1]),
            )

            save(fig, "heatmap_grade", group=group, save_directory=args.figdir)

            # Ngal
            fig, ax = plt.subplots(figsize=(15, 10))
            Simulation.heatmap_skymaps(
                ax,
                df[df["group"] == group]["simulation"].to_list(),
                df[df["group"] == args.heatmap]["simulation"].to_list(),
                tiles=np.array([5, 10, 70, -1]),
                attr="ngal_sum",
            )

            save(fig, "heatmap_ngal", group=group, save_directory=args.figdir)

    # Sun heatmap : the output must have multiple simulations with a telescope with "_sun" in its name and a single simulation with a telescope without "_sun".
    if args.sun:
        skymaps = np.unique(df["skymap"].to_numpy())
        for skymap in skymaps:
            print(f"Sun heatmap for : {skymap}")
            simus = df[df["skymap"] == skymap]
            tels = np.unique(simus["telescope"].to_numpy())
            if True in ["sun" in tel for tel in tels]:
                telescope_sun = tels[np.array(["sun" in tel for tel in tels])][0]
                fig, ax = plt.subplots(figsize=(15, 10))
                Simulation.heatmap_gps(
                    ax,
                    simus[simus["telescope"] == telescope_sun]["simulation"].to_list(),
                    simus[simus["telescope"] != telescope_sun]["simulation"].to_list()[
                        0
                    ],
                    iso=[0.5],
                )
                ax.tick_params(labelsize=20)

                save(fig, "heatmap_sun", group=skymap, save_directory=args.figdir)
