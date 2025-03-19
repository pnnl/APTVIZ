'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
'''

from flask import current_app as app
# from app import app
from flask import render_template, redirect, url_for, flash, get_flashed_messages, request
from utils.clustering import cluster_these_grids, count_clusters, get_kvals, get_iters
from utils.preprocess import process_raw
from utils.community_mapping import community_mapping
import os
import os.path as op
import json
from utils.file_handling import allowed_file
from werkzeug.utils import secure_filename
import networkx as nx


@app.route('/')
@app.route('/index')
def home():
    """
    Refreshes the config file and redirects to the home page.
    """
    config = {"filepath": "../data/2nm_gb_grid_radii_clustered.csv", "n_clusters": 5}
    with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
        json.dump(config, write_file)
    return render_template('index.html')

@app.route('/csv_uploads')
def csv_uploads():
    """
    reroutes to the page where users can upload their csv files.
    """
    return render_template('csv_uploads.html')

@app.route('/raw_uploads')
def raw_uploads():
    """
    reroutes to the page where users can upload their rrng and pos files (and soon apt files as well)
    """
    return render_template('raw_uploads.html')

@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    """
    Handles all file uploads and redirects to either the user-input selection page, or the visualization dashboard based on what was uploaded.
    """
    # access existing json config:
    with open(os.path.join(app.root_path, 'config.json'), 'r') as read_file:
        config = json.load(read_file)
    
    if request.method == 'POST':
        # check if post has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        #file = request.files['file']
        files = request.files.getlist("file") 

        for file in files:

            # if user does not select a file, browser submits an empty file without a filename
            if file.filename == '':
                flash("No selected file")
                return redirect(request.url)
            

            # if everything checks out we can save the file to the upload folder
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print('files are present and allowed. proceeding...')

                # csv and raw data redirect to different pages:
                if filename.rsplit('.', 1)[1].lower() == 'csv':
                    if 'yes' in request.form.getlist('isclustered'):
                        
                        # save the clustered csv and add its path to the json config:
                        clustered_filepath = op.join(app.root_path, 'data', filename)
                        file.save(clustered_filepath)

                        config['clustered_filepath'] = clustered_filepath
                        config['n_clusters'] = count_clusters(clustered_filepath)
                        config['k_vals'] = get_kvals(clustered_filepath)
                        config['n_iters'] = get_iters(clustered_filepath)

                        with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
                            json.dump(config, write_file)
                        return redirect("/dashboard/visualize-dashboard")
                    else:
                        # save the grids csv and add its path to the json config:
                        grids_filepath = op.join(app.root_path, 'data', filename)
                        file.save(grids_filepath)
                        config['grids_filepath'] = grids_filepath

                        with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
                            json.dump(config, write_file)

                        return redirect(url_for('visualize', name=filename))
                if filename.rsplit('.', 1)[1].lower() == 'pos':
                    print('pos file detected')
                    # save the pos file and add its path to the json config:
                    pos_filepath = op.join(app.root_path, 'data', filename)
                    file.save(pos_filepath)
                    config['pos_filepath'] = pos_filepath

                    with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
                        json.dump(config, write_file)
                if filename.rsplit('.', 1)[1].lower() == 'rrng':
                    print('rrng file detected')
                    # save the rrng file and add its path to the json config:
                    rrng_filepath = op.join(app.root_path, 'data', filename)
                    file.save(rrng_filepath)
                    config['rrng_filepath'] = rrng_filepath

                    with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
                        json.dump(config, write_file)
            else:
                # if there are no files, or they are not of allowed filetypes, redirect to the homepage
                print('a file with a non-allowed filetype was uploaded. please try again')
                return redirect(url_for('home', name=filename))
        
        # Once done saving all the uploaded files, check that all required parts are there
        with open(os.path.join(app.root_path, 'config.json'), 'r') as read_file:
            config = json.load(read_file)
        
        print(config.keys())
        if ('pos_filepath' in config.keys()) and ('rrng_filepath' in config.keys()):
            print('a pos and a rrng file were uploaded. processing')
            radius = 2
            overlap = 0.5
            sample = op.split(op.splitext(config['pos_filepath'])[0])[-1]
            grids_filename = f"{sample}_{radius}nm-radius_{overlap}-overlap.csv"
            grids_filepath = op.join(app.root_path, 'data', grids_filename)
            
            args = {'pos': config['pos_filepath'],
                    'rrng': config['rrng_filepath'],
                    'csv': grids_filepath,
                    'radius': radius,
                    'overlap': overlap}
            
            process_raw(args)
            
            # add the grids path to the json config:
            config['grids_filepath'] = grids_filepath

            with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
                json.dump(config, write_file)
            print('processing complete, moving to visualize page for clustering setting selection')
            return redirect(url_for('visualize', name=filename))
        
        # this currently shouldn't ever trigger
        print('something has gone wrong. please try again')
        return redirect(url_for('upload_file'))
    else:
        # the only way that this endpoint should be accessed is through the form. elsewise, this
        # will redirect it to the home page so the user can choose which type of upload they want to do
        print('the form input was not detected. try uploading again')
        return render_template('index.html')
    

@app.route('/testing_page', methods=["GET", "POST"])
def test_outputs():
    """
    reroutes to a testing page. this is unused unless testing a new feature.
    """
    return render_template('test_page.html')


@app.route('/visualize', methods=["GET", "POST"])
def visualize():
    """
    handles the form with user-selected inputs and clusters the data accordingly, then reroutes to the visualization dashboard. This route will also be used to
    create neighborhoods based on user selected parameters in the future.
    """

    # Processing clustering settings from the form then redirecting to the dash visualization
    if request.method == "POST":
        clusters_choice = request.form["clusters_choice"]
        seed = request.form["seed"]
        overlap = request.form["overlap"]
        method = request.form["method"]
        print(clusters_choice, seed, overlap, method)

        if seed:
            seed = int(seed)
        else:
            seed = 42

        if overlap:
            overlap = int(overlap)
        else:
            overlap = 50


        # access existing json config:
        with open(os.path.join(app.root_path, 'config.json'), 'r') as read_file:
            config = json.load(read_file)
        
        filepath = config['grids_filepath']


        if clusters_choice == "louvain":
            k_min = int(request.form["k_min"])
            k_max = int(request.form["k_max"])

            if k_min == k_max:
                kvals = [k_min]
            elif k_min < k_max:
                kvals = list(range(k_min, (k_max + 1)))
            else:
                # this is here in case the user puts the larger number in k_min
                kvals = list(range(k_max, (k_min + 1)))

            # can have users choose the number of repeats, but for now default is 10
            n_repeats = 10
            output_dict = community_mapping(filepath, kvals, n_repeats)
            n_clusters = output_dict['n_partitions']
            clustered_filepath = output_dict['clustered_filepath']
            config['pos'] = output_dict['pos']
            config['partition'] = output_dict['partition']
            config['G'] = output_dict['G']
            print(f'pos: type is {type(output_dict["pos"])}')
            print(f'partition: type is {type(output_dict["partition"])}')
            print(f'G: type is {type(output_dict["G"])}')

        else:
            n_clusters = int(request.form["n_clusters"])
            kvals = n_clusters
            n_repeats = 1
            clustered_filepath = cluster_these_grids(filepath, method, n_clusters, seed)

        
        # write to json config:
        config['clustered_filepath'] = clustered_filepath
        config['n_clusters'] = int(n_clusters)
        config['k_vals'] = kvals
        config['n_iters'] = n_repeats

        print(f"n_iters: {config['n_iters']}")
        print(f"kvals: {config['k_vals']}")

        with open(os.path.join(app.root_path, 'config.json'), 'w') as write_file:
            json.dump(config, write_file)
        return redirect("/dashboard/visualize-dashboard")
    return render_template('visualize.html')
