class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'casting':
            return 'C:\\Users\LMH\Desktop\personal\\f-AnoGan_mun\dataset\casting_data'  # folder that contains VOCdevkit/.
        elif database == 'test_air':
            return 'C:\\Users\LMH\Desktop\\air\\SO2_code3'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
