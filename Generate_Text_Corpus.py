#### 2017 - Jp mondoni - Sentiment Analysis of Amazon Book Reviews ####
## Simple code to generate one text file for each review on the UCI Dataset.
## Generated files will be used as text corpus for sentiment analysis on NLTK.

import csv, sys, os, time

rootFolder = 'C:\\Users\\joao.paulo\\Downloads\\corpora\\corpus\\'
bookList = ['WeirMartian', 'TarttGoldfinch', 'JamesShades', 
			'FillianGoneGirl','GreenStars', 'LauraUnbroken', 
			'HawkinsTrain', 'CollinsGames']
bookSum = len(bookList)
bookFileList = ['WeirMartian.txt', 'TarttGoldfinch.txt', 'JamesShades.txt', 
			'FillianGoneGirl.txt','GreenStars.txt', 'LauraUnbroken.txt', 
			'HawkinsTrain.txt', 'CollinsGames.txt']


def CreateReviewText(Sequence, BookId, Review):
	#print('Creating new text file') 
	name = rootFolder+str(BookId)+'\\'+str(BookId)+str(Sequence)+'.txt'  # Name of text file coerced with +.txt
	try:
		if not os.path.exists(rootFolder+str(BookId)+'\\'):
			os.makedirs(rootFolder+str(BookId)+'\\')
		txt = open(name,'w', encoding='utf-8')   # Trying to create a new file or open one
		print(str(Review), file=txt)
		txt.close()
	except  Exception as e:
		print(e)
		print('Failed to write the file '+str(BookId)+str(Sequence)+'.txt')
		print('tried to write to: '+ name)


i = 0
y = 0
rowlist=[]
for i in range(len(bookFileList)-1):
	print ('[' + time.strftime("%H:%M:%S") + ']> Generating corpus based on ' + bookList[i] + ' reviews.')
	with open(bookFileList[i], encoding='utf-8') as file:
		rowlist = file.readlines()
		rowlist = [x.strip() for x in rowlist]
		print('using ' + str(bookFileList[i]))
		n = 0
		for n in range(len(rowlist)):
			if n % 150 == 0:
				print('.', end='', flush=True)
			#print(rowlist[n])
			CreateReviewText(n, bookList[i], str(rowlist[n]))
			n=n+1
	i=i+1
	print('\nText corpus successfully generated with ' + str(n) + ' entries.')