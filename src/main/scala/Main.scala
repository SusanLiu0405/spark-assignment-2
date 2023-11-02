import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{sum, count}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}


object Main {
    def main(args: Array[String]): Unit = {
        // Initialize SparkSession
        val spark = SparkSession.builder()
            .appName("Main")
            .master("local[*]")
            .getOrCreate()

        // Load the training data
        val trainData: DataFrame = spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("src/main/scala/Titanic_Dataset/train.csv")

        // Load the testing data
        val testData: DataFrame = spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("src/main/scala/Titanic_Dataset/test.csv")


        // calculate the average ticket fare of each class
        val avgFarePerClass = trainData.groupBy("Pclass").agg(avg("Fare").as("Average_Fare"))
        avgFarePerClass.show()

        // Compute survival rate for each Pclass
        val survivalRatePerClass = trainData.groupBy("Pclass")
            .agg((sum("Survived") / count("Survived") * 100).alias("Survival_Rate_Percentage"))
            .orderBy(desc("Survival_Rate_Percentage"))

        // Print the survival rates
        survivalRatePerClass.show()

        val possibleRoses = trainData.filter(
            col("Age") === 17 &&
                col("Pclass") === 1 &&
                col("SibSp") === 0 &&
                col("Parch") === 1 &&
                col("Sex") === "female" &&
                col("Survived") === 1
        )
        // Count the possible Roses
        val numberOfPossibleRoses = possibleRoses.count()
        println(s"Number of possible Roses: $numberOfPossibleRoses")


        val possibleJacks = trainData.filter(
            col("Age") >= 19 && col("Age") < 21 &&
                col("Pclass") === 3 &&
                col("SibSp") === 0 &&
                col("Parch") === 0 &&
                col("Sex") === "male" &&
                col("Survived") === 0
        )

        // Count the possible Jacks
        val numberOfPossibleJacks = possibleJacks.count()
        println(s"Number of possible Jacks: $numberOfPossibleJacks")

        // split the group by age
        val ageGrouping = udf((age: Double) => {
            age match {
                case _ if age <= 10 => "1-10"
                case _ if age <= 20 => "11-20"
                case _ if age <= 30 => "21-30"
                case _ if age <= 40 => "31-40"
                case _ if age <= 50 => "41-50"
                case _ if age <= 60 => "51-60"
                case _ if age <= 70 => "61-70"
                case _ if age <= 80 => "71-80"
                case _ => "81+"
            }
        })
        val titanic_with_age_groups = trainData.withColumn("AgeGroup", ageGrouping(col("Age")))
        val age_fare_survival = titanic_with_age_groups.groupBy("AgeGroup")
            .agg(
                avg("Fare").alias("AverageFare"),
                (sum("Survived")/count("*") * 100).alias("SurvivalRate")
            )
            .orderBy("AgeGroup")
        age_fare_survival.show()

        // Do more statistics on "Age":
        val meanAge = trainData.agg(avg("Age")).first().getDouble(0)
        println(s"Average Age: $meanAge")
        val medianAge = trainData.stat.approxQuantile("Age", Array(0.5), 0.001)(0)
        println(s"Median Age: $medianAge")
        val modeAge = trainData.groupBy("Age").count().orderBy(desc("count")).limit(1).collect()(0)(0)
        println(s"Mode Age: $modeAge")


        //Find the columns that are sparse with data.
        // Get rid of some of the columns later!
        val nullCounts = trainData.columns.map { colName =>
            (colName, trainData.filter(trainData(colName).isNull || trainData(colName).isNaN).count())
        }

        val top5ColumnsWithMostNulls = nullCounts.sortBy(-_._2).take(5)

        println("Top 5 columns with most null values:")
        top5ColumnsWithMostNulls.foreach {
            case (column, count) => println(s"Column: $column, Null count: $count")
        }




        // Use training data & test data stats to fill null values
        val meanAgeTrain = trainData.agg(avg("Age")).first.getDouble(0)
        val meanFareTrain = trainData.agg(avg("Fare")).first.getDouble(0)
        val meanAgeTest = testData.agg(avg("Age")).first.getDouble(0)
        val meanFareTest = testData.agg(avg("Fare")).first.getDouble(0)

        // Also, drop the Cabin because of too many null values.
        // And Ticket is just a ticket number, which has nothing to do with our predictions
        val trainDataFilled = trainData.na.fill(Map("Age" -> meanAgeTrain, "Fare" -> meanFareTrain, "Embarked" -> "S")).drop("Cabin", "Ticket")
        val testDataFilled = testData.na.fill(Map("Age" -> meanAgeTest, "Fare" -> meanFareTest, "Embarked" -> "S")).drop("Cabin", "Ticket")

        // Indexing categorical features
        val catFeatColNames = Seq("Pclass", "Sex", "Embarked")
        val stringIndexers = catFeatColNames.map { colName =>
            new StringIndexer()
                .setInputCol(colName)
                .setOutputCol(colName + "Indexed")
                .setStringOrderType("frequencyDesc")
                .fit(trainDataFilled)
        }

        // Create new attributes: assembling features into one vector
        val numFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
        val idxdCatFeatColName = catFeatColNames.map(_ + "Indexed")
        val allIdxdFeatColNames = numFeatColNames ++ idxdCatFeatColName
        val assembler = new VectorAssembler()
            .setInputCols(allIdxdFeatColNames.toArray)
            .setOutputCol("Features")

        // split the trainDataFilled into trainingData and validationData
        // so that I could get the accuracy of the model
        val Array(trainingData, validationData) = trainDataFilled.randomSplit(Array(0.8, 0.2), seed = 1234)


        // RandomForest classifier
        val randomforest = new RandomForestClassifier()
            .setLabelCol("Survived")
            .setFeaturesCol("Features")

        // Creating pipeline
        val pipeline = new Pipeline().setStages(
            (stringIndexers :+ assembler :+ randomforest).toArray)

        val model = pipeline.fit(trainingData)

        // Predictions on validation data
        val predictions = model.transform(validationData)

        // Evaluating accuracy
        val evaluatorAccuracy = new MulticlassClassificationEvaluator()
            .setLabelCol("Survived")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")

        val accuracy = evaluatorAccuracy.evaluate(predictions)
        println(s"Validation Accuracy: ${accuracy * 100}%")

        val testPredictions = model.transform(testDataFilled)
        testPredictions.select("PassengerId", "prediction").show(10)
        testPredictions.select("PassengerId", "prediction").write.csv("src/main/scala/Titanic_Dataset/save_predictions.csv")
    }
}
