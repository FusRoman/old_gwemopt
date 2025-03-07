
import os, sys
import copy
import numpy as np
import healpy as hp
import gwemopt.plotting
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
from astropy import units as u

import gwemopt.utils, gwemopt.tiles
import gwemopt.rankedTilesGenerator
import gwemopt.scheduler
import ligo.segments as segments

def combine_coverage_structs(coverage_structs):

    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0,8))
    coverage_struct_combined["filters"] = np.empty((0,1))
    coverage_struct_combined["ipix"] = []
    coverage_struct_combined["patch"] = []
    coverage_struct_combined["FOV"] = np.empty((0,1))
    coverage_struct_combined["area"] = np.empty((0,1))
    coverage_struct_combined["telescope"] = np.empty((0,1))
    coverage_struct_combined["galaxies"] = []
    coverage_struct_combined["exposureused"] = []

    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = np.append(coverage_struct_combined["data"],coverage_struct["data"],axis=0)
        coverage_struct_combined["filters"] = np.append(coverage_struct_combined["filters"],coverage_struct["filters"])
        coverage_struct_combined["ipix"] = coverage_struct_combined["ipix"] + coverage_struct["ipix"]
        coverage_struct_combined["patch"] = coverage_struct_combined["patch"] + coverage_struct["patch"]
        coverage_struct_combined["FOV"] = np.append(coverage_struct_combined["FOV"],coverage_struct["FOV"])
        coverage_struct_combined["area"] = np.append(coverage_struct_combined["area"],coverage_struct["area"])
        coverage_struct_combined["telescope"] = np.append(coverage_struct_combined["telescope"],coverage_struct["telescope"])
        coverage_struct_combined["exposureused"] += list(coverage_struct["exposureused"])
        if "galaxies" in coverage_struct:
            coverage_struct_combined["galaxies"] = coverage_struct_combined["galaxies"] + coverage_struct["galaxies"]

    return coverage_struct_combined

def read_coverage(params, telescope, filename):

    nside = params["nside"]
    config_struct = params["config"][telescope]

    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    for line in lines:
        lineSplit = line.split(",")
        ra = float(lineSplit[2])
        dec = float(lineSplit[3])
        mjd = float(lineSplit[4])
        filt = lineSplit[6]
        mag = float(lineSplit[7])

        coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[ra,dec,mjd,mag,config_struct["exposuretime"],-1,-1,-1]]),axis=0)
        coverage_struct["filters"].append(filt)

        if telescope == "ATLAS":
            alpha=0.2
            color='#6c71c4'
        elif telescope == "PS1":
            alpha=0.1
            color='#859900'
        else:
            alpha=0.2
            color='#6c71c4'

        if config_struct["FOV_coverage_type"] == "square":
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra, dec, config_struct["FOV_coverage"], nside, alpha=alpha, color=color)
        elif config_struct["FOV_coverage_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra, dec, config_struct["FOV_coverage"], nside, alpha=alpha, color=color)

        coverage_struct["patch"].append(patch)
        coverage_struct["ipix"].append(ipix)
        coverage_struct["area"].append(area)

    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["FOV"] = config_struct["FOV_coverage"]*np.ones((len(coverage_struct["filters"]),))

    return coverage_struct

def read_coverage_files(params):

    coverage_structs = []
    for telescope, coverageFile in zip(params["telescopes"],params["coverageFiles"]):
        coverage_struct = read_coverage(params,telescope,coverageFile)
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)

def waw(params, map_struct, tile_structs): 

    nside = params["nside"]

    t = np.arange(0,7,1/24.0)
    #t = np.arange(0,7,1.0)
    cr90 = map_struct["cumprob"] < 0.9
    detmaps = gwemopt.waw.detectability_maps(params, t, map_struct, verbose=True, limit_to_region=cr90, nside=nside)

    coverage_structs = []
    for telescope in params["telescopes"]: 
        tile_struct = tile_structs[telescope]
        config_struct = params["config"][telescope]
        T_int = config_struct["exposuretime"]
        ranked_tile_probs = gwemopt.tiles.compute_tiles_map(tile_struct, map_struct["prob"], func='np.sum(x)')
        strategy_struct = gwemopt.waw.construct_followup_strategy_tiles(map_struct["prob"],detmaps,t,tile_struct,T_int,params["Tobs"])
        if strategy_struct is None:
            raise ValueError("Change distance scale...")
        strategy_struct = strategy_struct*86400.0
        keys = tile_struct.keys()
        for key, prob, exposureTime in zip(keys, ranked_tile_probs, strategy_struct):
            tile_struct[key]["prob"] = prob
            tile_struct[key]["exposureTime"] = exposureTime
            tile_struct[key]["nexposures"] = int(np.floor(exposureTime/config_struct["exposuretime"]))
        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    if params["doPlots"]:
        gwemopt.plotting.waw(params,detmaps,t,strategy_struct)

    return combine_coverage_structs(coverage_structs)

def powerlaw(params, map_struct, tile_structs,previous_coverage_struct=None):

    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["prob"]

    for jj, telescope in enumerate(params["telescopes"]):

        if params["doSplit"]:
            if "observability" in map_struct:
                map_struct["observability"][telescope]["prob"] = map_struct["groups"][n_scope]
            else:
                map_struct["prob"] = map_struct["groups"][n_scope]
            if n_scope < len(map_struct["groups"]) - 1:
                n_scope += 1
            else:
                n_scope = 0

        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        # Try to load the minimum duration of time from telescope config file
        # Otherwise set it to zero
        try:   
            min_obs_duration = config_struct["min_observability_duration"] / 24
        except:
            min_obs_duration = 0.0

        if "filt_change_time" in config_struct.keys(): filt_change_time = config_struct["filt_change_time"]
        else: filt_change_time = 0

        if params["doIterativeTiling"] and (params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.slice_galaxy_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doPerturbativeTiling"] and (jj>0) and (not params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.perturb_tiles(params, config_struct, telescope, map_struct_hold, tile_struct)
 
        if params["doOverlappingScheduling"]:
            tile_struct = gwemopt.utils.check_overlapping_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doAlternatingFilters"]:
            if params["doBlocks"]:
                tile_struct = gwemopt.utils.eject_tiles(params,telescope,tile_struct)
                   
            params_hold = copy.copy(params)
            config_struct_hold = copy.copy(config_struct)
            
            coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
            
            if params["doRASlices"] and params["do3D"]:

                coverage_struct = gwemopt.scheduler.schedule_ra_splits(params,config_struct,
                                                                       map_struct_hold,tile_struct,
                                                                       telescope,previous_coverage_struct)

  
            elif params["doRASlices"]:
                print("Need to enable --do3D if using --doRASlices")
                exit(0)
            elif params["doBalanceExposure"]:
                optimized_bool = False
                if not params["doMaxTiles"]: #optimize max tiles (iff max tiles not already specified)
                    optimized_bool = True
                    optimized_max,coverage_struct,tile_struct = gwemopt.utils.optimize_max_tiles(params,tile_struct,coverage_struct,config_struct,telescope,map_struct_hold)
                    params["max_nb_tiles"] = np.array([optimized_max],dtype=np.float)
                if not optimized_bool:
                    params_hold = copy.copy(params)
                    config_struct_hold = copy.copy(config_struct)
                    
                    coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
                    doReschedule,balanced_fields = gwemopt.utils.balance_tiles(params_hold, tile_struct, coverage_struct)

                    if doReschedule:
                        config_struct_hold = copy.copy(config_struct)
                        coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
                    
#                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params_hold,coverage_struct)
        else:

            #load the sun retriction for a satelite
            try:   
                sat_sun_restriction = config_struct["sat_sun_restriction"]
            except:
                sat_sun_restriction = 0.0
            
            if not params["tilesType"] == "galaxy":
                tile_struct = gwemopt.tiles.powerlaw_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct)  
    
            elif sat_sun_restriction == 0.0:
                for key in tile_struct.keys():
                    # Check that a given tile is observable a minimum amount of time
                    # If not set the proba associated to the tile to zero
                    if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist'] and min_obs_duration > 0.0:
                        observability_duration = 0.0
                        for counter in range(len(tile_struct[key]['segmentlist'])):
                            observability_duration += tile_struct[key]['segmentlist'][counter][1] - tile_struct[key]['segmentlist'][counter][0]
                        if tile_struct[key]['prob'] > 0.0 and observability_duration < min_obs_duration:
                            tile_struct[key]['prob'] = 0.0
            
            else:
                # Check that a given tile is not to close to the sun for the satelite
                # If it's to close set the proba associated to the tile to zero

                time = map_struct["trigtime"]
                time = Time(time, format='isot', scale='utc')
                sun_position = get_sun(time)
                
                #astropy don't like the output of get sun in the following separator function, need here to redefine the skycoord
                sun_position = SkyCoord(sun_position.ra, sun_position.dec, frame='gcrs') 
       
                for key in tile_struct.keys():      
                    if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist']:

                        for counter in range(len(tile_struct[key]['segmentlist'])):

                            tile_position = SkyCoord(ra=tile_struct[key]['ra']*u.degree, dec=tile_struct[key]['dec']*u.degree, frame='icrs')
                            ang_dist = sun_position.separation(tile_position).deg

                        if ang_dist < sat_sun_restriction:

                            tile_struct[key]['prob'] = 0.0

          
            if params["timeallocationType"] == "manual": #only works if using same telescope
                try:
                    for field_id in params["observedTiles"]:
                        field_id = int(field_id)
                        if field_id in tile_struct:
                            tile_struct[field_id]['prob'] = 0.0
            
                except:
                    raise ValueError("need to specify tiles that have been observed using --observedTiles")
        
            if (params["doUpdateScheduler"] or params["doTreasureMap"]) and previous_coverage_struct:
                tile_struct = update_observed_tiles(params,tile_struct,previous_coverage_struct) #coverage_struct of the previous round

            if params["doSuperSched"]:
                tile_struct = erase_observed_tiles(params,tile_struct,telescope)

            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

            if params["doBalanceExposure"]:
                cnt,ntrials = 0,10
                while cnt < ntrials:
                    doReschedule,balanced_fields = gwemopt.utils.balance_tiles(params, tile_struct, coverage_struct)
                    if doReschedule:
                        for key in params["unbalanced_tiles"]:
                            tile_struct[key]['prob'] = 0.0
                        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
                        cnt = cnt+1
                    else: break

#                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params,coverage_struct)

            if params["doMaxTiles"]:
                tile_struct, doReschedule = gwemopt.utils.slice_number_tiles(params, telescope, tile_struct, coverage_struct)    
                if doReschedule:
                    coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

        tile_structs[telescope] = tile_struct
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = gwemopt.utils.slice_map_tiles(params, map_struct_hold, coverage_struct)
               
    map_struct["prob"] = full_prob_map

    if params["doMovie_supersched"]:
        gwemopt.plotting.doMovie_supersched(params,combine_coverage_structs(coverage_structs),tile_structs,map_struct)
    
    return tile_structs, combine_coverage_structs(coverage_structs)

def absmag(params, map_struct, tile_structs,previous_coverage_struct=None):

    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["prob"]

    for jj, telescope in enumerate(params["telescopes"]):

        if params["doSplit"]:
            if "observability" in map_struct:
                map_struct["observability"][telescope]["prob"] = map_struct["groups"][n_scope]
            else:
                map_struct["prob"] = map_struct["groups"][n_scope]
            if n_scope < len(map_struct["groups"]) - 1:
                n_scope += 1
            else:
                n_scope = 0

        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        # Try to load the minimum duration of time from telescope config file
        # Otherwise set it to zero
        try:   
            min_obs_duration = config_struct["min_observability_duration"] / 24
        except:
            min_obs_duration = 0.0

        if "filt_change_time" in config_struct.keys(): filt_change_time = config_struct["filt_change_time"]
        else: filt_change_time = 0

        if params["doIterativeTiling"] and (params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.slice_galaxy_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doPerturbativeTiling"] and (jj>0) and (not params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.perturb_tiles(params, config_struct, telescope, map_struct_hold, tile_struct)
 
        if params["doOverlappingScheduling"]:
            tile_struct = gwemopt.utils.check_overlapping_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doAlternatingFilters"]:
            if params["doBlocks"]:
                tile_struct = gwemopt.utils.eject_tiles(params,telescope,tile_struct)
                   
            params_hold = copy.copy(params)
            config_struct_hold = copy.copy(config_struct)
            
            coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
            
            if params["doRASlices"] and params["do3D"]:

                coverage_struct = gwemopt.scheduler.schedule_ra_splits(params,config_struct,
                                                                       map_struct_hold,tile_struct,
                                                                       telescope,previous_coverage_struct)

  
            elif params["doRASlices"]:
                print("Need to enable --do3D if using --doRASlices")
                exit(0)
            elif params["doBalanceExposure"]:
                optimized_bool = False
                if not params["doMaxTiles"]: #optimize max tiles (iff max tiles not already specified)
                    optimized_bool = True
                    optimized_max = gwemopt.utils.optimize_max_tiles(params,tile_struct,coverage_struct,config_struct,telescope,map_struct_hold)
                    params["max_nb_tiles"] = np.array([optimized_max],dtype=np.float)

                params_hold = copy.copy(params)
                config_struct_hold = copy.copy(config_struct)
                
                coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
                doReschedule,balanced_fields = gwemopt.utils.balance_tiles(params_hold, tile_struct, coverage_struct)
                config_struct_hold = copy.copy(config_struct)

                if doReschedule:
                    coverage_struct,tile_struct = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct,previous_coverage_struct)
                
#                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params_hold,coverage_struct)
        else:

            #load the sun retriction for a satelite
            try:   
                sat_sun_restriction = config_struct["sat_sun_restriction"]
            except:
                sat_sun_restriction = 0.0
            
            if not params["tilesType"] == "galaxy":
                tile_struct = gwemopt.tiles.absmag_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct)  
    
            elif sat_sun_restriction == 0.0:
                for key in tile_struct.keys():
                    # Check that a given tile is observable a minimum amount of time
                    # If not set the proba associated to the tile to zero
                    if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist'] and min_obs_duration > 0.0:
                        observability_duration = 0.0
                        for counter in range(len(tile_struct[key]['segmentlist'])):
                            observability_duration += tile_struct[key]['segmentlist'][counter][1] - tile_struct[key]['segmentlist'][counter][0]
                        if tile_struct[key]['prob'] > 0.0 and observability_duration < min_obs_duration:
                            tile_struct[key]['prob'] = 0.0
            
            else:
                # Check that a given tile is not to close to the sun for the satelite
                # If it's to close set the proba associated to the tile to zero

                time = map_struct["trigtime"]
                time = Time(time, format='isot', scale='utc')
                sun_position = get_sun(time)
                
                #astropy don't like the output of get sun in the following separator function, need here to redefine the skycoord
                sun_position = SkyCoord(sun_position.ra, sun_position.dec, frame='gcrs') 
       
                for key in tile_struct.keys():      
                    if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist']:

                        for counter in range(len(tile_struct[key]['segmentlist'])):

                            tile_position = SkyCoord(ra=tile_struct[key]['ra']*u.degree, dec=tile_struct[key]['dec']*u.degree, frame='icrs')
                            ang_dist = sun_position.separation(tile_position).deg

                        if ang_dist < sat_sun_restriction:

                            tile_struct[key]['prob'] = 0.0

          
            if params["timeallocationType"] == "manual": #only works if using same telescope
                try:
                    for field_id in params["observedTiles"]:
                        field_id = int(field_id)
                        if field_id in tile_struct:
                            tile_struct[field_id]['prob'] = 0.0
            
                except:
                    raise ValueError("need to specify tiles that have been observed using --observedTiles")
        
            if (params["doUpdateScheduler"] or params["doTreasureMap"]) and previous_coverage_struct:
                tile_struct = update_observed_tiles(params,tile_struct,previous_coverage_struct) #coverage_struct of the previous round

            if params["doSuperSched"]:
                tile_struct = erase_observed_tiles(params,tile_struct,telescope)

            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

            if params["doBalanceExposure"]:
                cnt,ntrials = 0,10
                while cnt < ntrials:
                    doReschedule,balanced_fields = gwemopt.utils.balance_tiles(params, tile_struct, coverage_struct)
                    if doReschedule:
                        for key in params["unbalanced_tiles"]:
                            tile_struct[key]['prob'] = 0.0
                        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
                        cnt = cnt+1
                    else: break

#                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params,coverage_struct)

            if params["doMaxTiles"]:
                tile_struct, doReschedule = gwemopt.utils.slice_number_tiles(params, telescope, tile_struct, coverage_struct)    
                if doReschedule:
                    coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

        tile_structs[telescope] = tile_struct
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = gwemopt.utils.slice_map_tiles(params, map_struct_hold, coverage_struct)
               
    map_struct["prob"] = full_prob_map

    if params["doMovie_supersched"]:
        gwemopt.plotting.doMovie_supersched(params,combine_coverage_structs(coverage_structs),tile_structs,map_struct)
    
    return tile_structs, combine_coverage_structs(coverage_structs)

def pem(params, map_struct, tile_structs):

    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        if params["doAlternatingFilters"]:
            filters, exposuretimes = params["filters"], params["exposuretimes"]
            tile_struct_hold = copy.copy(tile_struct)
            coverage_structs_hold = []
            for filt, exposuretime in zip(filters,exposuretimes):
                params["filters"] = [filt]
                params["exposuretimes"] = [exposuretime]
                tile_struct_hold = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct_hold)
                coverage_struct_hold = gwemopt.scheduler.scheduler(params, config_struct, tile_struct_hold)
                coverage_structs_hold.append(coverage_struct_hold)
            coverage_struct = combine_coverage_structs(coverage_structs_hold)
        else:
            tile_struct = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct)
            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = gwemopt.utils.slice_map_tiles(map_struct_hold, coverage_struct)

    return combine_coverage_structs(coverage_structs)

def erase_observed_tiles(params,tile_struct,telescope): #only for run_gwemopt_superscheduler
    done_telescopes = []

    if len(params["coverage_structs"]) == 1: return tile_struct
    else:
        ii = len(params["coverage_structs"])-1 #finds out which round we are in

    while ii>0: #loops through all previous coverage structs + covered field ids to set tile probabilities to 0
        ii-=1
        prevtelescopes = params["alltelescopes"][ii].split(",")
        coverage_struct = params["coverage_structs"][f'coverage_struct_{ii}']
        if not coverage_struct: continue
        tile_struct_hold = gwemopt.utils.check_overlapping_tiles(params,tile_struct,coverage_struct)

        for prevtelescope in prevtelescopes:
            if prevtelescope in done_telescopes: continue #to prevent setting tiles to 0 redundantly
            done_telescopes.append(prevtelescope)

            if prevtelescope == telescope:
                for field_id in params["covered_field_ids"][prevtelescope][ii]:
                    tile_struct[field_id]['prob'] = 0.0
                continue

            for key in tile_struct.keys(): #maps field ids to tile_struct if not for the same telesocpe
                if not 'epochs' in tile_struct_hold[key]: continue
                epochs = tile_struct_hold[key]["epochs"]
                for jj in range(len(epochs)):
                    field_id = epochs[jj,5]
                    coverage_telescope = tile_struct[key]["epochs_telescope"][jj]
                    if field_id in params["covered_field_ids"][prevtelescope][ii] and coverage_telescope == prevtelescope: #makes sure field id obtained from check_overlapping_tiles is for the correct telescope
                        tile_struct[key]['prob'] = 0.0
                        break
    return tile_struct

def update_observed_tiles(params,tile_struct,previous_coverage_struct):
    
    if not params["doAlternatingFilters"]:
        tile_struct = gwemopt.utils.check_overlapping_tiles(params,tile_struct,previous_coverage_struct) #maps field ids to tile_struct

    for key in tile_struct.keys(): #sets tile to 0 if previously observed
        if 'epochs' not in tile_struct[key]: continue
        ipix = tile_struct[key]["ipix"]
        
        tot_overlap = sum(tile_struct[key]["epochs_overlap"]) #sums over list of overlapping ipix lengths

        if params["doAlternatingFilters"]:
            #only takes into account fields with same filters for total overlap
            for ii,filt in enumerate(tile_struct[key]["epochs_filters"]):
                if filt != params["filters"][0]:
                    tot_overlap -= tile_struct[key]["epochs_overlap"][ii]

        rat = tot_overlap/len(ipix)

        if rat > 0.3:
            tile_struct[key]['prob']=0.0

    return tile_struct

def timeallocation(params, map_struct, tile_structs,previous_coverage_struct=None):
    
    if len(params["telescopes"]) > 1 and params["doOrderByObservability"]:
        gwemopt.utils.order_by_observability(params,tile_structs)

    if (params["timeallocationType"] == "powerlaw") or (params["timeallocationType"] == "absmag"):
        print("Generating powerlaw schedule...")
        
        if params["doTreasureMap"]:
            if not params["treasuremap_token"]:
                print("Must provide Treasure Map API Token.")
                exit(0)
            treasuremap_coverage = gwemopt.utils.get_treasuremap_pointings(params)
        
            if previous_coverage_struct and treasuremap_coverage["data"]:
                previous_coverage_struct["data"] = np.append(previous_coverage_struct["data"],treasuremap_coverage["data"],axis=0)
                previous_coverage_struct["filters"] = previous_coverage_struct["filters"] + treasuremap_coverage["filters"]
                previous_coverage_struct["ipix"] = previous_coverage_struct["ipix"] + treasuremap_coverage["ipix"]
            elif treasuremap_coverage["data"]:
                previous_coverage_struct = treasuremap_coverage

        if (params["doUpdateScheduler"] or params["doTreasureMap"]) and not previous_coverage_struct:
                print("\nNo previous observations were ingested.\n")

        if params["doBlocks"]:
            exposurelists = {}
            scheduled_fields = {}
            for jj, telescope in enumerate(params["telescopes"]):
                config_struct = params["config"][telescope]
                exposurelist_split = np.array_split(config_struct["exposurelist"], params["Nblocks"])
                exposurelists[telescope] = exposurelist_split            
                scheduled_fields[telescope] = []
            tile_structs_hold = copy.copy(tile_structs)
            coverage_structs = []
            
            for ii in range(params["Nblocks"]):
                params_hold = copy.copy(params)
                params_hold["scheduled_fields"] = scheduled_fields
                for jj, telescope in enumerate(params["telescopes"]):
                    exposurelist = segments.segmentlist()
                    for seg in exposurelists[telescope][ii]:
                        exposurelist.append(segments.segment(seg[0],seg[1]))
                    params_hold["config"][telescope]["exposurelist"] = exposurelist

                    if params["timeallocationType"] == "absmag":
                        tile_structs_hold[telescope] = gwemopt.tiles.absmag_tiles_struct(params_hold, config_struct, telescope, map_struct, tile_structs_hold[telescope])
                    elif params["timeallocationType"] == "powerlaw":
                        tile_structs_hold[telescope] = gwemopt.tiles.powerlaw_tiles_struct(params_hold, config_struct, telescope, map_struct, tile_structs_hold[telescope])

                if params["timeallocationType"] == "absmag":
                    tile_structs_hold, coverage_struct = gwemopt.coverage.absmag(params_hold, map_struct, tile_structs_hold,previous_coverage_struct)
                elif params["timeallocationType"] == "powerlaw":
                    tile_structs_hold, coverage_struct = gwemopt.coverage.powerlaw(params_hold, map_struct, tile_structs_hold,previous_coverage_struct)

                coverage_structs.append(coverage_struct)
                for ii in range(len(coverage_struct["ipix"])):
                    telescope = coverage_struct["telescope"][ii]
                    scheduled_fields[telescope].append(coverage_struct["data"][ii,5]) #appends all scheduled fields to appropriate list

            coverage_struct = combine_coverage_structs(coverage_structs)
        else:
            if params["timeallocationType"] == "absmag":
                tile_structs, coverage_struct = gwemopt.coverage.absmag(params, map_struct, tile_structs,previous_coverage_struct)
            elif params["timeallocationType"] == "powerlaw":
                tile_structs, coverage_struct = gwemopt.coverage.powerlaw(params, map_struct, tile_structs,previous_coverage_struct)

    elif params["timeallocationType"] == "waw":
        if params["do3D"]:
            print("Generating WAW schedule...")
            coverage_struct = gwemopt.coverage.waw(params, map_struct, tile_structs)
        else:
            raise ValueError("Need to enable --do3D for waw")

    elif params["timeallocationType"] == "manual":
        print("Generating manual schedule...")
        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(params, map_struct, tile_structs)

    elif params["timeallocationType"] == "pem":
        print("Generating PEM schedule...")
        coverage_struct = gwemopt.coverage.pem(params, map_struct, tile_structs)

    return tile_structs, coverage_struct 
