# Талья Роман 203Б
import pathlib as pl
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error

names = ['avgT', 'minT', 'maxT', 'avgHum', 'avgDew', 'avgPrec', 'avgP', 'avgW', 'avgTM']
predictors = ['avgT-1', 'minT-1', 'maxT-1', 'avgDew-1', 'avgHum-1', 'avgPrec-1', 'avgP-1',
              'avgT-2', 'minT-2', 'maxT-2', 'avgDew-2', 'avgHum-2', 'avgPrec-2', 'avgP-2']
dataset = read_csv(pl.Path("weatherKazan2021.csv"), parse_dates=['date'], dayfirst=True,
                   names=['date'] + names + predictors, delimiter=";")
while True:
    try:
        ans = int(input("What correlation do you wish to see? Input a number from 1 to 9. (chooses from the "
                        "\"names\" array) To exit enter 0.\n"))
    except ValueError:
        continue
    if ans > 0:
        print(dataset.corr()[[names[ans - 1]]].sort_values(names[ans - 1]))
    else:
        break

indexes = [1, 4, 7]
print("The values with Pearson's correlation index of 0.6 and more are taken as predictors (aside from current"
      "day values, such as maxT, minT, avgT etc.\n")
listsOfPredictors = [['avgTM', 'avgT-1', 'minT-1', 'maxT-1', 'avgDew-1', 'avgHum-1', 'avgT-2',
                      'minT-2', 'maxT-2', 'avgDew-2', 'avgHum-2'],
                     ['avgTM', 'maxT-1', 'avgDew-1', 'maxT-2', 'avgDew-2'],
                     ['avgP-1', 'avgP-2']]
model = LinearRegression()
arr = dataset.values
listOfData = []
for i in range(3):
    XT = dataset[listsOfPredictors[i]]
    y = arr[:, indexes[i]]
    print('Predicting %s...' % names[indexes[i] - 1])
    XT_train, XT_valid, y_train, y_valid = train_test_split(XT, y, test_size=0.20, random_state=10)
    model.fit(XT_train, y_train)
    predictions = model.predict(XT_valid)
    formatPred = []
    for i in predictions:
        formatPred.append(round(i, 1))
    print('Accuracy of prediction: ', round(model.score(XT_valid, y_valid), 2) * 100, '%')
    print("The Mean Absolute Error: %.2f" % mean_absolute_error(y_valid, predictions))
    print("The Median Absolute Error: %.2f" % median_absolute_error(y_valid, predictions))
    while True:
        ans = input("Do you wish to see the predictions? y/n\n")
        if ans == 'y':
            print('Comparison:')
            print('Prediction VS Reality')
            i = 0
            while i < len(formatPred):
                print(formatPred[i], ' VS ', y_valid[i])
                i += 1
            input("Press any key to continue.\n")
            break
        if ans == 'n':
            break
    listOfData.append(formatPred)
    listOfData.append(y_valid)

print("Graph comparison:")
pyplot.subplot(3, 1, 1)
guess = pyplot.scatter([i for i in range(len(listOfData[0]))], listOfData[0], color="red")
real = pyplot.scatter([i for i in range(len(listOfData[1]))], listOfData[1], color="black")
pyplot.xlabel("Index")
pyplot.ylabel("Average temperature")
pyplot.grid("both", linewidth=1)
pyplot.xticks([i for i in range(len(listOfData[0]))], labels=[])
pyplot.legend([guess, real], ['предсказания программы', 'реальные данные'], bbox_to_anchor=(0.4, 1.1))

pyplot.subplot(3, 1, 2)
guess = pyplot.scatter([i for i in range(len(listOfData[2]))], listOfData[2], color="red")
real = pyplot.scatter([i for i in range(len(listOfData[3]))], listOfData[3], color="black")
pyplot.xlabel("Index")
pyplot.ylabel("Average humidity")
pyplot.grid("both", linewidth=1)
pyplot.xticks([i for i in range(len(listOfData[2]))], labels=[])

pyplot.subplot(3, 1, 3)
guess = pyplot.scatter([i for i in range(len(listOfData[4]))], listOfData[4], color="red")
real = pyplot.scatter([i for i in range(len(listOfData[5]))], listOfData[5], color="black")
pyplot.xlabel("Index")
pyplot.ylabel("Average pressure")
pyplot.grid("both", linewidth=1)
pyplot.xticks([i for i in range(len(listOfData[4]))], labels=[])

pyplot.show()
