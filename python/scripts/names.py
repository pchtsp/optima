import re


def simulation_params_fr():
    return {
        'num_resources': "nombre total de ressources. En ressources."
        , 'num_parallel_tasks': "nombre total de tâches en parallèle. En tâches."
        , 'maint_duration': "durée de la maintenance. En périodes."
        , 'max_used_time': "butée horaire. En heures."
        , 'max_elapsed_time': "butée calendaire. En périodes."
        , 'elapsed_time_size': "avance maximale pour déclencher la maintenance. En périodes."
        , 'min_usage_period': "quantité minimale de consommation des ressources. En heures."
        , 'perc_capacity': "taille de l'atelier. En pourcentage de la taille de la flotte."
        , 'min_avail_percent': "quantité minimale d'avions disponibles par cluster. En pourcentage de la taille du cluster."
        , 'min_avail_value': "quantité minimale d'avions disponibles par cluster. En ressources."
        , 'min_hours_perc': "quantité minimale d'heures totales de chaque ensemble de cluster. En pourcentage de la taille du cluster."
        , 'seed': None
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': "options de besoins d'affectation minimale des tâches. En périodes."
        , 'initial_unbalance': "intervalle possible de différence entre la butée calendaire et horaire initiales des ressources. En périodes."
        , 't_required_hours': "options des besoins d'heures des tâches. En heures."
        , 't_num_resource': "options des besoins de ressources des tâches. En ressources."
        , 't_duration': "intervalle de durée possible des tâches. En périodes."
        , 'perc_in_maint': "quantité de ressources en maintenance au début de l'horizon. En pourcentage de la taille de la flotte."
    }


def names_no_spaces():
    names = simulation_params_fr()
    return {n: re.sub('_', '', n) for n in names.keys()}


def names_latex():
    return {
        'num_resources': "$\mid I \mid$"
        , 'num_parallel_tasks': "$\mid J \mid$"
        , 'maint_duration': "$M$"
        , 'max_used_time': "$H^{M}$"
        , 'max_elapsed_time': "$E^{m}$"
        , 'elapsed_time_size': "$E^{s}$"
        , 'min_usage_period': "$R^{min}_j$"
        , 'perc_capacity': "$C^{perc}$"
        # TODO:
        , 'min_avail_percent': ""
        , 'min_avail_value': ""
        , 'min_hours_perc': ""
    }
