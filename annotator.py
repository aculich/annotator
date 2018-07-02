from IPython.core.display import display, HTML
from ipywidgets import Layout
import ipywidgets as widgets
import requests, re, numpy as np

# For the inline classifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

class Annotator:
	experiment = ""
	labels = []

	final_choice = ""
	current_query = None
	ipython = False
	annotator = ''
	printify = lambda x: str(x)
	start_labels = []	
	access = ""
	web_root = ""
	shuffle = True
	classifier = None
	use_classifier = False
	classifier_trained = False

	def __init__(self, experiment, printify=None, ipython=False, access='web',
				 web_root='', start_labels=[], annotator='',
				 shuffle=True, use_classifier=False, close_choice_mode=False, choice_function=None):
		# access can be:
		# 	- File: and provide a file (file) to the experiment
		# 	- Web: and provide a URL (web_root) to the Flask server with API hooks

		# printify is a callable that takes in a sample and produces a string to print the sample to the user
		# start_labels: initial set of possible labels, can be updated over time
		# shuffle: If try, the samples are annotated in a random order? Default True
		# use_classifier: if true will train a classifier every few samples and then suggests the classes in order

		# close_choice_mode: if the choices available are known in advance, must provide `choice_function`
		# choice_function: if close_choice_mode is chosen, then a function that goes from a sample to the list of displayed choices

		if printify is not None:
			self.printify = printify
		self.experiment = experiment
		self.ipython = ipython
		self.access = access
		self.web_root = web_root
		self.start_labels = start_labels
		self.annotator = annotator # Name of the person annotating
		self.shuffle = shuffle

		self.use_classifier = use_classifier
		self.classifier_trained = False
		if self.use_classifier:
			self.classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
		
		self.close_choice_mode = close_choice_mode
		self.choice_function = choice_function

	def clean_html(self, raw_html):
		cleanr = re.compile('<.*?>')
		return re.sub(cleanr, '', raw_html)

	def query2json(self, query):
		url = self.web_root+query
		# print url
		r = requests.get(url)
		return r.json()

	# Web api hooks: 7 functions
	def list_experiments_web(self):
		results = self.query2json("annotator/list_experiments")
		for d in results:
			print "Experiment:", d['_id'], "has", d['count'], "samples"

	def insert_samples_web(self, samples):
		R = requests.post(self.web_root+"annotator/insert_samples", json=samples)
		print R.text

	def save_annotation_web(self):
		print "hellllo"
		self.query2json("annotator/save_annotation/"+str(self.current_query['_id'])+"/label/"+self.final_choice+"/annotator/"+self.annotator)

	def get_counts_web(self):
		return self.query2json("annotator/get_counts/"+self.experiment+"/annotator/"+self.annotator)

	def load_example_web(self):
		return self.query2json("annotator/next_example/"+self.experiment+"/annotator/"+self.annotator+"/shuffle/"+str(1 if self.shuffle else 0))

	def reload_labels_web(self):
		self.labels = self.query2json("annotator/reload_labels/"+self.experiment)

	def detailed_statistics_web(self):
		return self.query2json("annotator/detailed_statistics/"+self.experiment)

	def relabel_web(self, from_label, to_label):
		return self.query2json("annotator/relabel/"+self.experiment+"/from/"+from_label+"/"+to_label+"/annotator/"+self.annotator)

	def load_labeled_dataset_web(self):
		return self.query2json("annotator/load_training/"+self.experiment)

	def export_web(self):
		return self.query2json("annotator/export/"+self.experiment)

	# Main functionality
	def list_experiments():
		if self.access == 'web':
			self.list_experiments_web()

	def insert_samples(self, samples, pre_annotated=False):
		for sample in samples:
			if type(sample) is not dict:
				print "One sample or more is not a dict, which it should be. Please reformat"
				return -1
			if 'label' in sample and not pre_annotated:
				print "The key `label` should not be used in the samples"
				return -2
			if 'experiment' in sample and sample['experiment'] != self.experiment:
				print "The key `experiment` is inconsistent with the experiment ID"
				return -3
			sample['experiment'] = self.experiment
		if self.access == 'web':
			self.insert_samples_web(samples)

	def save_annotation(self):
		if self.access == 'web':
			self.save_annotation_web()

	def load_unannotated_example(self):
		if self.access == 'web':
			self.current_query = self.load_example_web()

	def reload_labels(self):
		if self.access == 'web':
			self.reload_labels_web()

		self.labels.extend(self.start_labels)
		self.labels = sorted(self.labels)

	def relabel(self, from_label, to_label):
		if self.access == 'web':
			self.relabel_web(from_label, to_label)

	def export(self):
		if self.access == 'web':
			return self.export_web()

	def finish(self):
		# Print a message once we are done
		print "Done annotating for now"

	def train_classifier(self):
		labeled_dataset = []
		if self.access == 'web':
			labeled_dataset = self.load_labeled_dataset_web()
		if len(labeled_dataset) > 20:
			labels = [d['label']['label'] for d in labeled_dataset]
			text = [self.clean_html(self.printify(d)) for d in labeled_dataset]

			cross_val_accuracy = np.mean(cross_val_score(self.classifier, text, y=labels, scoring="accuracy", cv=5))
			print "Classifier retrained (",len(labeled_dataset)," samples). Cross val accuracy:", "{0:.2f}%".format(100.0*cross_val_accuracy)

			self.classifier.fit(text, y=labels)

			self.classifier_trained = True
		else:
			self.classifier_trained = False

	def get_ordered_labels(self):
		# Current example has been loaded, get the label either through the classifier
		# or take the first label
		if self.close_choice_mode:
			first_choice = ""
			text_labels = self.choice_function(self.current_query)
			if len(text_labels) > 0:
				first_choice = text_labels[0]
			return text_labels, first_choice

		if self.use_classifier:
			if not self.classifier_trained:
				self.train_classifier()
			if self.classifier_trained:
				best_label = self.classifier.predict([self.clean_html(self.printify(self.current_query))])[0]
				scores = self.classifier.decision_function([self.clean_html(self.printify(self.current_query))])[0]
				zero_min = (scores - np.min(scores))
				normalized_scores = zero_min / np.sum(zero_min)
				labels = self.classifier.named_steps['clf'].classes_
				lab2p = {lab: score for lab, score in zip(labels, normalized_scores)}
				sorted_labels = sorted(labels, key=lambda x: -lab2p[x]) + [labs for labs in self.start_labels if labs not in labels]
				text_labels = [lab+" | Score: "+"{0:.1f}".format(100.0*lab2p.get(lab, 0.0)) for lab in sorted_labels]

				return text_labels, text_labels[0]

		best_label = ""
		if len(self.labels) > 0:
			best_label = self.labels[0]

		return self.labels, best_label

	def load_example_ipython(self):

		self.load_unannotated_example()
		if self.current_query is None:
			self.finish()
			return None # We stop here

		if not self.close_choice_mode:
			self.reload_labels() # In case something new has been added...

		display(HTML(self.printify(self.current_query)))

		TextField = widgets.Text(value='', placeholder='New class label', disabled=False)
		TextField.observe(self.on_change_jupyter)

		text_labels, self.final_choice = self.get_ordered_labels()

		Radio = widgets.RadioButtons(options=text_labels, value=self.final_choice, description="", disabled=False)
		Radio.observe(self.on_change_jupyter)

		B = widgets.Button(description='Submit annotation')
		B.on_click(self.on_submit_jupyter)

		count_annotated, count_total = self.get_counts()
		if count_annotated % 10 == 0:
			self.classifier_trained = False # Force retrain
		display(HTML("<div class='toDel'>"+str(count_annotated)+"/"+str(count_total)+"</div>"))
		display(TextField)
		display(Radio)
		display(B)

	def cleanup_jupyter(self):
		display(HTML("<div class='js_stuff'><script>$('#example, .js_stuff').parent().parent().remove(); $('#desc_rows, .output_area, .toDel').remove(); $('.widget-subarea').html('');</script></div>"))

	def on_change_jupyter(self, change):
		if change['type'] == 'change' and change['name'] == 'value':
			self.final_choice = change['new']

	def on_submit_jupyter(self, change):
		# Save this annotation

		print "coucouchou"
		self.final_choice = self.final_choice.split("|")[0].strip()

		print self.final_choice
		self.save_annotation()
		self.cleanup_jupyter()
		self.load_example_ipython()

	def annotate(self):
		if self.ipython:
			self.load_example_ipython()

	# Analysis tools
	def get_counts(self):
		if self.access == 'web':
			return self.get_counts_web()
		return 0, 0

	def status(self):
		count_annotated, count_total = self.get_counts()
		return "["+self.experiment+"] Total samples: "+ str(count_total) + " | Annotated: "+ str(count_annotated)

	def detailed_statistics(self):
		count_annotated, count_total = self.get_counts()

		for d in self.detailed_statistics_web():
			percentage = "{0:.2f}".format(100.0*d['count']/count_annotated)
			print d['_id'], d['count'], " / ", count_annotated, " (", percentage ," % )"

if __name__ == "__main__":
	anno = Annotator("test")
	anno.load_example_ipython()