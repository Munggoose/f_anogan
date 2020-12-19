class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return 'E:\VOCdevkit\VOC2012'  # folder that contains VOCdevkit/.
        elif database == 'test_air':
            return 'C:\\Users\LMH\Desktop\\air\\SO2_code3'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
