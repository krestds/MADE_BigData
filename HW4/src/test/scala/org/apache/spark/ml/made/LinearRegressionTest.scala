package org.apache.spark.ml.made

import breeze.linalg.DenseVector
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val weights:Vector  = LinearRegressionTest._weights

  "Model" should "check prediction with const weights" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      stds = weights
    ).setInputCol("features")
      .setOutputCol("label")

    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double]("label"))

    vectors.length should be(2)

    vectors(0) should be(0.9538140387469525 +- delta)
    vectors(1) should be(1.051468826282612 +- delta)
  }
  //@Ignore
  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double]("label"))

    vectors.length should be(2)

    vectors(0) should be(0.9538140387469525 +- delta)
    vectors(1) should be(1.051468826282612 +- delta)
  }

  //@Ignore

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val vectors: Array[Double] = reRead.transform(data).collect().map(_.getAs[Double]("label"))

    vectors.length should be(2)

    vectors(0) should be(0.9538140387469525 +- delta)
    vectors(1) should be(1.051468826282612 +- delta)

  }

}

object LinearRegressionTest extends WithSpark {

  lazy val _weights = Vectors.dense(0.030457152388021148, 1.0807994102404535, 0.364472609163103, 0.39165989017724295)

  lazy val _vectors = Seq(
    Vectors.dense(0.3745401173314895, 0.5951556213611038, 0.10181166516532092, 0.6690886967898408),
    Vectors.dense(0.9507143077766118, 0.3647171449244542, 0.2982834771509757, 1.326688171136571)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}