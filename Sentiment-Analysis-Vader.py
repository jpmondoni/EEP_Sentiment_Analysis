## Algorithm to read reviews from dataset and generate csv with results.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv, time, os
from nltk.tokenize import word_tokenize

#Windows Google Drive Folder Tree
#rootFolder = 'C:\\Users\\joao.paulo\\Google Drive\\TCC\\Desenvolvimento\\''
#dataFolder = 'Datasets\\amazon_book_reviews\\parsed\\'
#codeFolder = 'Code\\''
#resultFoder = 'generated\\'

#Linux Google Drive Folder Tree
rootFolder = '/home/jp/Google Drive/TCC/Desenvolvimento/'
dataFolder = 'Datasets/amazon_book_reviews/parsed/'
codeFolder = 'Code/'
resultFolder = 'generated/'
review_set = []
sentiment_set = []
csvFilePath = rootFolder + dataFolder + resultFolder + fileName[:-4] + '-scores.csv'

bookList = ['The Martian', 'The Goldfinch', '50 Shades of Grey', 
			'Gone Girl','The Fault in our Stars', 'Unbroken', 
			'The girl on the Train', 'Hunger Games','Snippet - Hunger Games']
bookFileList = ['WeirMartian.txt', 'TarttGoldfinch.txt', 'JamesShades.txt', 
			'FillianGoneGirl.txt','GreenStars.txt', 'LauraUnbroken.txt', 
			'HawkinsTrain.txt', 'CollinsGames.txt', 'HungerSnippet.txt']


print('Choose book to generate csv file: ')
print('	1. The Martian')
print('	2. The Goldfinch')
print('	3. 50 Shades of Grey')
print('	4. Gone Girl')
print('	5. The Fault in our Stars')
print('	6. Unbroken')
print('	7. The Girl on the Train')
print('	8. Hunger Games')
print('	9. Snippet - Hunger Games')

chIndex = input("$ ")
chIndex = int(chIndex)
fileName = bookFileList[(chIndex-1)]

with open(rootFolder + dataFolder + fileName, encoding='utf-8') as f:
	for review in f:
		review_set.append(review)


def generateCSV(reviewList):
	with open(csvFilePath, 'a') as reviewsCSV: # Append mode
		for i in range(len(reviewList)):
			wr = csv.writer(reviewsCSV)
			wr.writerow(reviewList[i])

analyzer = SentimentIntensityAnalyzer()
print('[' + time.strftime("%H:%M:%S") + ']$ Generating sentiment analysis based on ' + bookList[chIndex-1] + ' ' + str(len(review_set)) + ' reviews.')

for review in review_set:
		vs = analyzer.polarity_scores(review)
		#print("{:-<65} {}". format(review, str(vs)))
		# Tokenize string to get score values by index and append on list, as: compound, pos, neu, neg and review text.
		score = word_tokenize(str(vs))
		neg_score = score[4]
		neu_score = score[9]
		pos_score = score[14]
		compound = score[19]
		sentiment_set.append([compound, pos_score, neu_score, neg_score, review])

generateCSV(sentiment_set)
print ('[' + time.strftime("%H:%M:%S") + ']$ Results file (csv) successfully generated at ' + csvFilePath)
