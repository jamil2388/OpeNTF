import pandas as pd
from time import time
from tqdm import tqdm
import os
import multiprocessing
from math import isnan

from cmn.team import Team
from cmn.inventor import Inventor

import pickle

class Patent(Team):
    def __init__(self, id, members, date, title, country, subgroups, withdrawn, members_details):
        super().__init__(id, members, set(subgroups.split(',')), date)
        self.title = title
        self.country = country
        self.subgroups = subgroups
        self.withdrawn = withdrawn
        self.members_details = members_details

        for i, member in enumerate(self.members):
            member.teams.append(self.id)
            member.skills.update(set(self.skills))
            member.locations.append(self.members_details[i])

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        try:
            return super(Patent, Patent).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} ...")

            #data dictionary can be find at: https://patentsview.org/download/data-download-dictionary
            print("Reading patents ...")
            patents = pd.read_csv(datapath, sep='\t', header=0, dtype={'id':'object'}, usecols=['id', 'type', 'country', 'date', 'title', 'withdrawn'], low_memory=False)#withdrawn may imply success or failure
            patents.rename(columns={'id': 'patent_id', 'country':'patent_country'}, inplace=True)
            patents = patents[patents['type'].isin(['utility', ''])]

            print("Reading patents' subgroups ...")
            patents_cpc = pd.read_csv(datapath.replace('patent', 'cpc_current'), sep='\t', dtype={'patent_id':'object'}, usecols=['patent_id', 'subgroup_id', 'sequence'])
            patents_cpc.sort_values(by=['patent_id', 'sequence'], inplace=True)#to keep the order of subgroups
            patents_cpc.reset_index(drop=True, inplace=True)
            patents_cpc = patents_cpc.groupby(['patent_id'])['subgroup_id'].apply(','.join).reset_index()
            patents_cpc = pd.merge(patents, patents_cpc, on='patent_id', how='inner', copy=False)

            #TODO: filter the patent based on subgroup e.g., cpc_subgroup: "Y10S706/XX"	"Data processing: artificial intelligence"

            print("Reading patents' inventors ...")
            patents_inventors = pd.read_csv(datapath.replace('patent', 'patent_inventor'), sep='\t', header=0, dtype={'patent_id':'object'})
            patents_cpc_inventors = pd.merge(patents_cpc, patents_inventors, on='patent_id', how='inner', copy=False)

            print("Reading inventors ...")
            inventors = pd.read_csv(datapath.replace('patent', 'inventor'), sep='\t', header=0, dtype={'male_flag':'boolean'}, usecols=['id', 'name_first', 'name_last', 'male_flag'])
            patents_cpc_inventors = pd.merge(patents_cpc_inventors, inventors, left_on='inventor_id', right_on='id', how='inner', copy=False)
            patents.rename(columns={'id': 'inv_id'}, inplace=True)

            print("Reading location data ...")
            locations = pd.read_csv(datapath.replace('patent', 'location'), sep='\t', header=0, usecols=['id', 'city', 'state', 'country'])
            patents_cpc_inventors_location = pd.merge(patents_cpc_inventors, locations, left_on='location_id', right_on='id', how='inner', copy=False)

            patents_cpc_inventors_location.sort_values(by=['patent_id'], inplace=True)
            patents_cpc_inventors_location = patents_cpc_inventors_location.append(pd.Series(), ignore_index=True)

            print("Reading data to objects..")
            teams = {}; candidates = {}; n_row = 0
            current = None

            # 100 % |██████████████████████████████████████████████████████████████████████▉ | 210808 / 210809[00:02 < 00:00,75194.06 it / s]
            for patent_team in tqdm(patents_cpc_inventors_location.itertuples(), total=patents_cpc_inventors_location.shape[0]):
                try:
                    if pd.isnull(new := patent_team.patent_id): break
                    if current != new:
                        team = Patent(patent_team.patent_id,#for "utility" patents is integer but for "design" has "Dxxxx", ...
                                      [],
                                      patent_team.date,
                                      patent_team.title,
                                      patent_team.patent_country,
                                      patent_team.subgroup_id,
                                      bool(patent_team.withdrawn),
                                      [])
                        current = new
                        teams[team.id] = team

                    inventor_id = patent_team.inventor_id
                    inventor_name = f'{patent_team.name_first}_{patent_team.name_last}'

                    if (idname := f'{inventor_id}_{inventor_name}') not in candidates:
                        candidates[idname] = Inventor(patent_team.inventor_id, inventor_name, patent_team.male_flag)
                    team.members.append(candidates[idname])
                    team.members_details.append((patent_team.city, patent_team.state, patent_team.country))

                    candidates[idname].skills.update(team.skills)
                    candidates[idname].teams.append(team.id)
                    candidates[idname].locations.append(team.members_details[-1])

                except Exception as e:
                    raise e
            return super(Patent, Patent).read_data(teams, output, filter, settings)

    @classmethod
    def get_stats(cls, teams, teamsvecs, output, plot=False):
        try:
            print("Loading the stats pickle ...")
            with open(f'{output}/stats.pkl', 'rb') as infile:
                print('Found the file')
                stats = pickle.load(infile)

                if plot: Team.plot_stats(stats, output)
                return stats

        except FileNotFoundError:
            print('Stats.pkl File Not Found. Re-Generating Stats.')
            stats = {}
            t = time()
            stats.update(super().get_stats(teamsvecs, output, plot))
            os.remove(f'{output}/stats.pkl')
            print('Deleted Redundant File')
            print(f'Time Taken to do Stats from Super Class {(time() - t) / 60} Minutes')
            t1 = time()
            t_country = 0;
            unq_country = set()
            total_patents = len(teams.keys())
            for key in tqdm(teams.keys()):
                loc = teams[key].members_details
                for item in loc:
                    _, _, country_name = item
                    t_country = t_country + 1
                    unq_country.add(country_name)

            stats['navg_inv_country_per_patent'] = t_country / total_patents

            unq_country = {x for x in unq_country if x == x}
            stats['unique_countries'] = unq_country
            stats['nunique_country'] = len(unq_country)

            max_records = teamsvecs['id'].shape[0]
            country_mem = {}
            for i in tqdm(range(0, max_records)):
                id = teamsvecs['id'][i].astype(int).toarray()[0][0].tolist()
                loc = teams[f'{id}'].members_details[0:]
                for loc_i in loc:
                    _, _, country_name = loc_i
                    if country_name in country_mem.keys():
                        country_mem[country_name] = country_mem[country_name] + 1
                    else:
                        country_mem[country_name] = 1

            country_skill = {}
            for idx, idr in enumerate(teams.keys()):
                for _, x in enumerate(teamsvecs['skill'][idx]):
                    skills = len(x.data[0])
                for rec in teams[idr].members_details:
                    _, _, country = rec
                    if country not in country_skill.keys():
                        country_skill[country] = skills
                    else:
                        country_skill[country] = country_skill[country] + skills

            stats['nskills_country-idx'] = country_skill
            stats['nmembers_country-idx'] = country_mem
            with open(f'{output}/stats.pkl', 'wb') as outfile:
                pickle.dump(stats, outfile)

            print(f'Time Taken to do complete Stats and save as pickle file {(time() - t1) / 60} Minutes')
            if plot: Team.plot_stats(stats, output)
            return stats
