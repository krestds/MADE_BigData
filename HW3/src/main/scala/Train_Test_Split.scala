import breeze.linalg.DenseMatrix

object Train_Test_Split {
  def apply(x:DenseMatrix[Double], y:DenseMatrix[Double], testSize:Double=.3, seed:Int = 42): (
    DenseMatrix[Double],
      DenseMatrix[Double],
      DenseMatrix[Double],
      DenseMatrix[Double]) = {

    val Num_rows = (x.rows * testSize).toInt
    val test_row = Get_Rows_Number(Num_rows, x.rows, seed)
    val train_row = (
      for (i <- (0 to x.rows - 1)
           if !test_row.contains(i)
           )
      yield i)

    val X_train: DenseMatrix[Double] = x(train_row , ::).toDenseMatrix
    val X_test: DenseMatrix[Double] = x(test_row, ::).toDenseMatrix
    val y_train: DenseMatrix[Double] = y(train_row , ::).toDenseMatrix
    val y_test: DenseMatrix[Double] = y(test_row, ::).toDenseMatrix

    (
      X_train,
      y_train,
      X_test,
      y_test
    )
  }

  def Get_Rows_Number(num:Int, count:Int, seed:Int): IndexedSeq[Int] ={
    val r = new scala.util.Random
    r.setSeed(seed)

    var seq: Seq[Int] = Seq()
    while (seq.size <= num){
      val rand = r.nextInt(count)
      if (!seq.contains(rand)){
        seq = seq ++ Seq(rand)
      }}
    seq.sorted.toIndexedSeq
  }
}
