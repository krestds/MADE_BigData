import breeze.linalg._
import java.io._

object Main extends App {
  val conf = new Conf(args)

  val csvFile = csvread(
    new File(conf.train()),
    ',',
    skipLines=1
  )

  val cols = (0 to csvFile.cols - 2)

  val x = csvFile(::, cols).toDenseMatrix
  val y = csvFile(::, IndexedSeq(csvFile.cols - 1)).toDenseMatrix

  val (x_train, y_train, x_test, y_test) = Train_Test_Split(x, y)

  val linearRegression = new LinearRegression()
  linearRegression.fit(x_train, y_train)

  val y_predict = linearRegression.predict(x_test)
  val mse = linearRegression.mse(y_predict, y_test)

  val pw = new PrintWriter(new File("validation.txt" ))
  pw.write(s"train mse: $mse")
  pw.close



  val test = csvread(
    new File(conf.test()),
    ',',
    skipLines=1
  )

  val result = linearRegression.predict(test)

  csvwrite(
    new File(conf.output()),
    result,
    ','
  )

}

