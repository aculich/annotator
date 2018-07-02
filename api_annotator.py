from newslens_api import app, db
from bson import ObjectId
from flask import request
import json

@app.route("/api/annotator/list_experiments")
def annotator_list_experiments():
	return db.annotation.aggregate([{'$sortByCount': '$experiment'}])

@app.route("/api/annotator/insert_samples", methods=["POST"])
def annotator_insert_samples():
	data = json.loads(request.data)
	if all(['experiment' in x for x in data]):
		db.annotation.insert_many(data) # This is not safe
		return "Samples added to experiment"
	else:
		return "Error occurred"

@app.route("/api/annotator/save_annotation/<sid>/label/<label>/annotator/<annotator>")
def annotator_save_annotation(sid, label, annotator):
	db.annotation.update_one({'_id': ObjectId(sid)}, {'$push': {"label": {"label": label, 'annotator': annotator}}})
	return ["Annotation saved"]

@app.route("/api/annotator/get_counts/<experiment>/annotator/<annotator>")
def annotator_get_counts(experiment, annotator):
	return [db.annotation.find({"label.annotator": annotator, 'experiment': experiment}).count(),
			db.annotation.find({'experiment': experiment}).count()]

@app.route("/api/annotator/next_example/<experiment>/annotator/<annotator>/shuffle/<shuffle>")
def annotator_next_example(experiment, annotator, shuffle):
	shuffle = True if shuffle == '1' else False
	pipeline = [{'$match': {"label.annotator": {'$ne': annotator}, 'experiment': experiment}}]
	if shuffle:
		pipeline.append({'$sample': {'size': 1}})
	else:
		pipeline.append({'$sort': {'_id': 1}})
		pipeline.append({'$limit': 1})

	sample = list(db.annotation.aggregate(pipeline))
	if len(sample) == 0:
		return None
	return sample[0]

@app.route("/api/annotator/reload_labels/<experiment>")
def annotator_reload_labels(experiment):
	return list(db.annotation.distinct("label.label", {'experiment': experiment}))

@app.route("/api/annotator/export/<experiment>")
def annotator_export(experiment):
	return list(db.annotation.find({'experiment': experiment}))

@app.route("/api/annotator/detailed_statistics/<experiment>")
def annotator_detailed_statistics(experiment):
	return list(db.annotation.aggregate([{'$match': {'experiment': experiment, 'label': {'$exists': True}}}, {'$sortByCount': "$label"}]))

@app.route("/api/annotator/relabel/<experiment>/from/<from_label>/<to_label>")
def annotator_relabel(experiment, from_label, to_label):
	db.annotation.update_many({'experiment': experiment, 'label': from_label}, {'$set': {'label': to_label}})
	return ["Finished relabeling"]

@app.route("/api/annotator/load_training/<experiment>")
def annotator_load_training(experiment):
	return list(db.annotation.aggregate([{"$match": {'experiment': experiment, 'label': {'$exists': True}}}, {'$unwind': '$label'}]))