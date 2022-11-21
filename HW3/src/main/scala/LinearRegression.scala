import breeze.linalg.{DenseMatrix, inv}
import breeze.numerics.{sqrt, pow}

class LinearRegression {
  var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](0, 0)

  def fit(x: DenseMatrix[Double], y:DenseMatrix[Double]): Unit = {
    val x_modif = modifyData(x)
    w = inv(x_modif.t * x_modif) * x_modif.t * y
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val y: DenseMatrix[Double] = modifyData(x) * w
    y
  }

  def modifyData(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val zeros = DenseMatrix.ones[Double](x.rows, 1)
    DenseMatrix.horzcat(x, zeros)
  }

  def mse(y: DenseMatrix[Double], y_pred: DenseMatrix[Double]):Double = {
    val array = pow(y - y_pred, 2).toArray
    val mse = array.sum / array.length
    mse
  }

}
