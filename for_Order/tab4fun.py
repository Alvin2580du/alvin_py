import pandas as pd
import numpy as np

#
data_train = pd.read_csv("./datasets/tab4fun/tap_fun_train.csv",
                         # usecols=['prediction_pay_price'],
                         usecols=['user_id', 'wood_add_value', 'wood_reduce_value', 'stone_add_value',
                                  'stone_reduce_value', 'ivory_add_value', 'ivory_reduce_value', 'meat_add_value',
                                  'meat_reduce_value', 'magic_add_value', 'magic_reduce_value', 'infantry_add_value',
                                  'infantry_reduce_value', 'cavalry_add_value', 'cavalry_reduce_value',
                                  'shaman_add_value', 'shaman_reduce_value', 'wound_infantry_add_value',
                                  'wound_infantry_reduce_value', 'wound_cavalry_add_value',
                                  'wound_cavalry_reduce_value', 'wound_shaman_add_value', 'wound_shaman_reduce_value',
                                  'general_acceleration_add_value', 'general_acceleration_reduce_value',
                                  'building_acceleration_add_value', 'building_acceleration_reduce_value',
                                  'reaserch_acceleration_add_value', 'reaserch_acceleration_reduce_value',
                                  'training_acceleration_add_value', 'training_acceleration_reduce_value',
                                  'treatment_acceleraion_add_value', 'treatment_acceleration_reduce_value',
                                  'bd_training_hut_level', 'bd_healing_lodge_level', 'bd_stronghold_level',
                                  'bd_outpost_portal_level', 'bd_barrack_level', 'bd_healing_spring_level',
                                  'bd_dolmen_level', 'bd_guest_cavern_level', 'bd_warehouse_level',
                                  'bd_watchtower_level', 'bd_magic_coin_tree_level', 'bd_hall_of_war_level',
                                  'bd_market_level', 'bd_hero_gacha_level', 'bd_hero_strengthen_level',
                                  'bd_hero_pve_level', 'sr_scout_level', 'sr_training_speed_level',
                                  'sr_infantry_tier_2_level', 'sr_cavalry_tier_2_level', 'sr_shaman_tier_2_level',
                                  'sr_infantry_atk_level', 'sr_cavalry_atk_level', 'sr_shaman_atk_level',
                                  'sr_infantry_tier_3_level', 'sr_cavalry_tier_3_level', 'sr_shaman_tier_3_level',
                                  'sr_troop_defense_level', 'sr_infantry_def_level', 'sr_cavalry_def_level',
                                  'sr_shaman_def_level', 'sr_infantry_hp_level', 'sr_cavalry_hp_level',
                                  'sr_shaman_hp_level', 'sr_infantry_tier_4_level', 'sr_cavalry_tier_4_level',
                                  'sr_shaman_tier_4_level', 'sr_troop_attack_level', 'sr_construction_speed_level',
                                  'sr_hide_storage_level', 'sr_troop_consumption_level', 'sr_rss_a_prod_levell',
                                  'sr_rss_b_prod_level', 'sr_rss_c_prod_level', 'sr_rss_d_prod_level',
                                  'sr_rss_a_gather_level', 'sr_rss_b_gather_level', 'sr_rss_c_gather_level',
                                  'sr_rss_d_gather_level', 'sr_troop_load_level', 'sr_rss_e_gather_level',
                                  'sr_rss_e_prod_level', 'sr_outpost_durability_level', 'sr_outpost_tier_2_level',
                                  'sr_healing_space_level', 'sr_gathering_hunter_buff_level', 'sr_healing_speed_level',
                                  'sr_outpost_tier_3_level', 'sr_alliance_march_speed_level',
                                  'sr_pvp_march_speed_level', 'sr_gathering_march_speed_level',
                                  'sr_outpost_tier_4_level', 'sr_guest_troop_capacity_level', 'sr_march_size_level',
                                  'sr_rss_help_bonus_level', 'pvp_battle_count', 'pvp_lanch_count', 'pvp_win_count',
                                  'pve_battle_count', 'pve_lanch_count', 'pve_win_count', 'avg_online_minutes',
                                  'pay_price', 'pay_count', 'prediction_pay_price'],
                         dtype=np.float16)
print(data_train.shape)
data_train_pay = data_train[data_train.prediction_pay_price > 0]

# corr = []

corr = data_train.corr()
corr.to_csv('./datasets/tab4fun/corr.csv', index=None)

corrdata_train_pay = data_train_pay.corr()
corrdata_train_pay.to_csv('./datasets/tab4fun/data_train_paycorr.csv', index=None)
