import org.rogach.scallop.ScallopConf

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train= opt[String](required = true, short = 'i')
  val test= opt[String](required = true, short = 't')
  val output = opt[String](required = true, short = 'o')
  verify()

  val config = Map(
    "train" -> (if (train.isEmpty) train.toString() else Nil),
    "test" -> (if (test.isEmpty) test.toString() else Nil),
    "output" -> (if (output.isEmpty) output.toString() else Nil)
  )
}