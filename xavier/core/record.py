import time
from pathlib import Path

from xavier.lib.cortex.cortex import Cortex


class Record:
    def __init__(self, app_client_id, app_client_secret, record_export_folder, **kwargs):
        self.record_title = 'values'
        self.record_description = ""
        self.record_export_data_types = ['BP']
        self.record_export_format = 'CSV'
        self.record_export_version = 'V2'
        self.record_id = None
        self.record_export_folder = record_export_folder

        Path(self.record_export_folder).mkdir(parents=True, exist_ok=True)

        self.c = Cortex(app_client_id, app_client_secret, debug_mode=False, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(create_record_done=self.on_create_record_done)
        self.c.bind(stop_record_done=self.on_stop_record_done)
        self.c.bind(warn_cortex_stop_all_sub=self.on_warn_cortex_stop_all_sub)
        self.c.bind(export_record_done=self.on_export_record_done)
        self.c.bind(inform_error=self.on_inform_error)

    def start(self):
        print("Start Record")
        self.c.open()

    def stop(self):
        print("Stop Record")
        self.c.stop_record()

    def export_record(self, folder, stream_types, export_format, record_ids, version, **kwargs):
        self.c.export_record(folder, stream_types, export_format, record_ids, version, **kwargs)

    def on_create_session_done(self, *args, **kwargs):
        self.c.create_record(self.record_title, description=self.record_description)

    def on_create_record_done(self, *args, **kwargs):
        data = kwargs.get('data')
        self.record_id = data['uuid']

    def on_stop_record_done(self, *args, **kwargs):
        self.c.disconnect_headset()

    def on_warn_cortex_stop_all_sub(self, *args, **kwargs):
        time.sleep(3)

        self.export_record(self.record_export_folder,
                           self.record_export_data_types,
                           self.record_export_format,
                           [self.record_id],
                           self.record_export_version)

    def on_export_record_done(self, *args, **kwargs):
        self.c.close()
        print("File saved")

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(error_data)
        try:
            if self.c.session_id != '':
                self.c.stop_record()
                self.c.close()
        except Exception as ex:
            print(ex)
            raise KeyboardInterrupt
