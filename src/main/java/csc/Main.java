package csc;

import opennlp.tools.cmdline.CmdLineUtil;
import opennlp.tools.cmdline.TerminateToolException;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.lemmatizer.*;
import opennlp.tools.postag.*;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.*;
import opennlp.tools.util.eval.FMeasure;
import opennlp.tools.util.model.ModelUtil;

import java.io.*;
import java.nio.charset.Charset;

public class Main {

    public static void main3(String[] args) {
        TrainingParameters mlParams = new TrainingParameters();
        mlParams.put("Iterations", 10);

        InputStreamFactory inputStreamFactory = null;
        try {
            inputStreamFactory = new MarkableFileInputStreamFactory(
                    new File("C:\\Users\\CX70\\Documents\\opennlp-rus\\train-lemma.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ObjectStream<String> lineStream = null;
        LemmaSampleStream lemmaStream = null;
        try {
            lineStream = new PlainTextByLineStream(
                    (inputStreamFactory), "UTF-8");
            lemmaStream = new LemmaSampleStream(lineStream);
        } catch (IOException e) {
            CmdLineUtil.handleCreateObjectStreamError(e);
        }

        LemmatizerModel model;
        try {
            LemmatizerFactory lemmatizerFactory = LemmatizerFactory
                    .create(null);
            model = LemmatizerME.train("ru", lemmaStream, mlParams,
                    lemmatizerFactory);
        } catch (IOException e) {
            throw new TerminateToolException(-1,
                    "IO error while reading training data or indexing data: "
                            + e.getMessage(),
                    e);
        } finally {
            try {
                lemmaStream.close();
            } catch (IOException e) {
            }
        }
        OutputStream modelOut = null;
        try {
            modelOut = new BufferedOutputStream(new FileOutputStream("trained-lemma.bin"));
            model.serialize(modelOut);
        } catch (IOException e) {
            // Failed to save model
            e.printStackTrace();
        } finally {
            if (modelOut != null) {
                try {
                    modelOut.close();
                } catch (IOException e) {
                    // Failed to correctly save model.
                    // Written model might be invalid.
                    e.printStackTrace();
                }
            }
        }
    }


    public static void main(String[] args) throws IOException {
        InputStreamFactory inputStreamFactory = null;
        try {
            inputStreamFactory = new MarkableFileInputStreamFactory(
                    new File("C:\\Users\\CX70\\Documents\\opennlp-rus\\test-lemma.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ObjectStream<String> lineStream = null;
        LemmaSampleStream lemmaStream = null;
        try {
            lineStream = new PlainTextByLineStream(
                    (inputStreamFactory), "UTF-8");
            lemmaStream = new LemmaSampleStream(lineStream);
        } catch (IOException e) {
            CmdLineUtil.handleCreateObjectStreamError(e);
        }

        LemmatizerModel model;
        int overall = 0, correct = 0;
        try (InputStream modelIn = new FileInputStream("trained-lemma.bin")) {
            model = new LemmatizerModel(modelIn);
        }
        LemmatizerME lemmatizer = new LemmatizerME(model);
        LemmaSample samp = null;
        while((samp = lemmaStream.read()) != null) {
            try {
                String[] res = lemmatizer.lemmatize(samp.getTokens(), samp.getTags());
                if (res[0].equals(samp.getLemmas()[0])) {
                    correct++;
                }

            }
            catch(Exception e) {overall++;}
        }
        System.out.println(correct);
        System.out.println(overall);
    }

    public static void main2(String[] args) {
        POSModel model = null;

        InputStream dataIn = null;
        try {
            Charset charset = Charset.forName("UTF-8");
            InputStreamFactory isf = new MarkableFileInputStreamFactory(new File("demofile.txt"));
            InputStreamFactory isf2 = new MarkableFileInputStreamFactory(new File("demofile-test.txt"));


            TrainingParameters mlParams = new TrainingParameters();
            mlParams.put("TrainerType", "Event");
            mlParams.put("Cutoff", 5);
            String[] algo = {"MAXENT"};
            Integer[] iters = {500};

            Tokenizer tokenizer = WhitespaceTokenizer.INSTANCE;


            for(String alg: algo) {
                for(Integer it: iters) {
                    mlParams.put("Algorithm", alg);
                    mlParams.put("Iterations", it);
                    ObjectStream<String> lineStream = new PlainTextByLineStream(isf, "UTF-8");
                    ObjectStream<POSSample> sampleStream = new WordTagSampleStream(lineStream);
                    model = POSTaggerME.train("ru", sampleStream, mlParams, new POSTaggerFactory());
                    POSTaggerME tagger = new POSTaggerME(model);
                    POSSample sample;
                    Measure measure = new Measure();
                    ObjectStream<String> lineStream2 = new PlainTextByLineStream(isf2, "UTF-8");
                    ObjectStream<POSSample> sampleStream2 = new WordTagSampleStream(lineStream2);
                    while((sample = sampleStream2.read()) != null) {
                        String[] pred = tagger.tag(sample.getSentence());
                        measure.update(pred, sample.getTags());
                    }
                    System.out.println(String.format("alg %s iter %s fm %s prec %s recall %s acc %s", alg, it, measure.getF1(), measure.getPrecision(), measure.getRecall(), measure.getAccuracy()));
                }
            }
        } catch (IOException e) {
            // Failed to read or parse training data, training failed
            e.printStackTrace();
        } finally {
            if (dataIn != null) {
                try {
                    dataIn.close();
                } catch (IOException e) {
                    // Not an issue, training already finished.
                    // The exception should be logged and investigated
                    // if part of a production system.
                    e.printStackTrace();
                }
            }
        }

        OutputStream modelOut = null;
        try {
            modelOut = new BufferedOutputStream(new FileOutputStream("trained.bin"));
            model.serialize(modelOut);
        } catch (IOException e) {
            // Failed to save model
            e.printStackTrace();
        } finally {
            if (modelOut != null) {
                try {
                    modelOut.close();
                } catch (IOException e) {
                    // Failed to correctly save model.
                    // Written model might be invalid.
                    e.printStackTrace();
                }
            }
        }
    }
}
