import os

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        # ---- Local path fix ----
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        datasets_path = os.path.join(BASE_DIR, 'datasets')
        return os.path.join(datasets_path, database.upper())