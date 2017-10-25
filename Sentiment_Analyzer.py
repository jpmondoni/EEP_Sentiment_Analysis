from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
import collections, time, pickle, csv, sys
from random import randint
from featx import bag_of_words, bag_of_words_in_set, train_test_feats
from ToneAnalyzerAPI import analyze_tone, display_results
from sklearn import cross_validation

####################
# Setup            #
####################

# Folder Tree
rootFolder = ''
dataFolder = ''
codeFolder = 'Code/'
resultFolder = 'results/'

# Initial Lists
review_set = []
sentiment_set = []
csvFilePath = ''

bookList = ['The Martian', 'The Goldfinch', '50 Shades of Grey', 
			'Gone Girl','The Fault in our Stars', 'Unbroken', 
			'The girl on the Train', 'Hunger Games','Snippet - Hunger Games', 'Test Review']
bookFileList = ['WeirMartian.txt', 'TarttGoldfinch.txt', 'JamesShades.txt', 
			'FillianGoneGirl.txt','GreenStars.txt', 'LauraUnbroken.txt', 
			'HawkinsTrain.txt', 'CollinsGames.txt', 'HungerSnippet.txt']

# NLTK Stopwords
sw = stopwords.words('english')


def generateCSV(reviewList, path):
	with open(path, 'a', encoding = 'utf-8') as reviewsCSV: # Append mode
		for i in range(len(reviewList)):
			wr = csv.writer(reviewsCSV)
			wr.writerow(reviewList[i])


def label_feats_from_corpus(corp, feature_detector=bag_of_words):
	label_feats = collections.defaultdict(list)
	for label in corp.categories():
		for fileid in corp.fileids(categories=[label]):
			feats = feature_detector(corp.words(fileids=[fileid]))
			label_feats[label].append(feats)
	return label_feats


def split_label_feats(lfeats, split=0.75):
	train_feats = []
	test_feats = []
	for label, feats in lfeats.items():
		cutoff = int(len(feats) * split)
		train_feats.extend([(feat, label) for feat in feats[:cutoff]])
		test_feats.extend([(feat, label) for feat in feats[cutoff:]])
	return train_feats, test_feats

####################
# Vader Analyzer   #
####################
def Vader(review_set, csvFilePath, chIndex):
	analyzer = SentimentIntensityAnalyzer()
	print('[' + time.strftime("%H:%M:%S") + ']$ Generating sentiment analysis based on ' + bookList[chIndex-1] + '\'s ' + str(len(review_set)) + ' review(s).')
	neg_c = pos_c = neu_c = i = 0
	for review in review_set:
		sentence_list = sent_tokenize(review)
		avg_compound = paragraphSentiments=0.0
		for sentence in sentence_list:
			vs = analyzer.polarity_scores(sentence)
			print("{:-<69} {}".format(sentence, str(vs["compound"])))
			paragraphSentiments += vs["compound"]
		avg_compound = paragraphSentiments/len(sentence_list)
		if(avg_compound >= 0.5):
			sentiment = 'pos'
		elif(avg_compound < 0.5 and avg_compound > 0.1):
			sentiment = 'slightly_pos'
		elif(avg_compound < 0.1 and avg_compound > -0.1):
					sentiment = 'neu'
		elif(avg_compound < -0.1 and avg_compound > -0.5):
			sentiment = 'slightly_neg'
		elif(avg_compound < -0.5):
			sentiment = 'neg'	

		avg_compound = round(avg_compound, 4)
		sentiment_set.append([sentiment, avg_compound, review[:-1]])
	print ('[' + time.strftime("%H:%M:%S") + ']$ Writing results file (csv) at ' + csvFilePath)
	generateCSV(sentiment_set, csvFilePath)
	print ('[' + time.strftime("%H:%M:%S") + ']$ Results file (csv) successfully generated at ' + csvFilePath)


####################
# Bayes Analyzer   #
####################
def NaiveBayes(review_set, csvFilePathNltk, chIndex):
	print('[' + time.strftime("%H:%M:%S") + ']$ Starting NLTK Naive Bayes Classifier Sentiment Analyzer...')

	movie_reviews.categories()

	lfeats = label_feats_from_corpus(movie_reviews)
	lfeats.keys()
	train_feats, test_feats = split_label_feats(lfeats, split=0.75)
	cv = cross_validation.KFold(len(train_feats), n_folds=10, shuffle=True, random_state=None)
	for traincv, evalcv in cv:
		nb_classifier = NaiveBayesClassifier.train(train_feats[traincv[0]:traincv[len(traincv)-1]])
		save_classifier = open("Classifier-CV.pickle","wb")
		pickle.dump(nb_classifier, save_classifier)
		save_classifier.close()

	print('[' + time.strftime("%H:%M:%S") + ']$ Generating NLTK sentiment analysis based on ' + bookList[chIndex-1] + '\'s ' + str(len(review_set)) + ' review(s).')
	sentiment_set = []
	for review in review_set:
		filtered_review = review
		for word in sw:
			filtered_review = filtered_review.replace(" " + word + " ", " ")

		diff_sw = len(filtered_review) / len(review)
		feats = bag_of_words(word_tokenize(filtered_review))
		sentiment = nb_classifier.classify(feats)
		probs = nb_classifier.prob_classify(feats)
		pos_prob = round(probs.prob('pos'), 4)
		neg_prob = round(probs.prob('neg'), 4)
		neu_prob = round(pos_prob - neg_prob, 4)
		pct_red = round(100 - (100*diff_sw), 2)

		sentiment_set.append([sentiment, pos_prob, neg_prob, review[:-1]])
		
	if(chIndex != 10):
		print('[' + time.strftime("%H:%M:%S") + ']$ Writing NLTK results file (csv) at ' + csvFilePathNltk)
		generateCSV(sentiment_set, csvFilePathNltk)
		print('[' + time.strftime("%H:%M:%S") + ']$ Results file (csv) successfully generated at ' + csvFilePathNltk)
	else:
		print(sentiment, pos_prob, neg_prob, neu_prob, filtered_review, pct_red)

	print('[' + time.strftime("%H:%M:%S") + ']$ Finished NLTK Naive Bayes Classifier Sentiment Analyzer!')
  
  
####################
# Main Program   #
####################
def main():
	# args: -S (submit text)/-B (books) -W (watson) -R (Random Watson)
	# se usado -S, nao pode ser usado o -R!


	if sys.argv[1] == "-S":
		test_text = input("Enter text for analysis: ")

		review_set.append(test_text)
		Vader(review_set, '',10)
		print('--------------')
		NaiveBayes(review_set, '', 10)

	elif sys.argv[1] == "-B":
		print('Choose book to generate csv file: ')
		print('	 1. The Martian')
		print('	 2. The Goldfinch')
		print('	 3. 50 Shades of Grey')
		print('	 4. Gone Girl')
		print('	 5. The Fault in our Stars')
		print('	 6. Unbroken')
		print('	 7. The Girl on the Train')
		print('	 8. Hunger Games')
		print('	 9. Snippet - Hunger Games')
		print('	66. Generate review for all books.')

		chIndex = input("$ ")
		chIndex = int(chIndex)
		if(chIndex == 66):
			for i in range(1, 10):
				del review_set[:]
				fileName = bookFileList[(i-1)]
				csvFilePathVader = rootFolder + dataFolder + resultFolder + fileName[:-4] + '-vader.csv'
				csvFilePathNltk = rootFolder + dataFolder + resultFolder + fileName[:-4] + '-nltk.csv'

				with open(rootFolder + dataFolder + fileName, encoding='utf-8') as f:
							for review in f:
								review_set.append(review)
				print('--------------')
				Vader(review_set, csvFilePathVader, i)
				print('--------------')
				NaiveBayes(review_set, csvFilePathNltk, i)	

		else:
			fileName = bookFileList[(chIndex-1)]
			csvFilePathVader = rootFolder + dataFolder + resultFolder + fileName[:-4] + '-vader.csv'
			csvFilePathNltk = rootFolder + dataFolder + resultFolder + fileName[:-4] + '-nltk.csv'	
			with open(rootFolder + dataFolder + fileName, encoding='utf-8') as f:
				for review in f:
					review_set.append(review)

			Vader(review_set, csvFilePathVader, chIndex)
			print('--------------')
			NaiveBayes(review_set, csvFilePathNltk, chIndex)

  ####################
  # Bluemix Args     #
  ####################
	if(len(sys.argv) > 2):	
		if sys.argv[2] == "-W":	
			if sys.argv[3] == "-R":
				rnd = randint(0,len(review_set)-1)
				data = review_set[rnd]
			else:
				data = test_text

			results = analyze_tone(data)
			print(analyze_tone(data))
			display_results(results)
					



main()
