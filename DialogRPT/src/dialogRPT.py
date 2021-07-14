from DialogRPT.src.model import OptionInfer, JointScorer, Scorer

def getIntegrated():
    path_ranker = "DialogRPT/restore/ensemble.yml"
    opt = OptionInfer()
    ranker = JointScorer(opt = opt)
    ranker.load(path_ranker)
    from DialogRPT.src.generation import GPT2Generator, Integrated
    cuda = True
    chkpoint = 'DialoGPT/output'
    generator = GPT2Generator(chkpoint, cuda)
    generator.predict = generator.predict_beam
    return Integrated(generator, ranker)
    