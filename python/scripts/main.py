import os, sys
import importlib
import argparse
import datetime as dt
import json
import package.exec as exec
import desktop_app.app as gui_app
import data.data_input as di

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-c', dest='file', default="package.params",
                        help='config file (default: package.params)')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-df', '--options-file', dest='config_file')
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)
    parser.add_argument('-it', '--input-template', dest='input_template')
    parser.add_argument('-id', '--input-template-dir', dest='input_template_dir')
    parser.add_argument('-gui', '--desktop-app', dest='open_desktop_app', action='store_true')

    args = parser.parse_args()
    print('Using config file in {}'.format(args.file))
    params = importlib.import_module(args.file)

    args = parser.parse_args()
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        root = sys._MEIPASS + '/'
    else:
        # we are running in a normal Python environment
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'

    params.OPTIONS['root'] = root

    new_options = None
    if args.config_dict:
        new_options = args.config_dict
    elif args.config_file:
        new_options = di.load_data(args.config_file)
    if new_options:
        params.OPTIONS.update(new_options)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)
        path = os.path.join(params.PATHS['experiments'], dt.datetime.now().strftime("%Y%m%d%H%M")) + '/'
        exec.update_case_path(params.OPTIONS, path)

    if args.input_template_dir:
        exec.udpdate_case_read_options(params.OPTIONS, args.input_template_dir)

    if args.input_template:
        params.OPTIONS['input_template_path'] = args.input_template

    options = params.OPTIONS
    options['PATHS'] = params.PATHS
    if args.open_desktop_app:
        gui_app.MainWindow_EXCEC(options)
    else:
        exec.config_and_solve(options)