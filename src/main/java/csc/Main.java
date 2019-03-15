package csc;

import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.postag.*;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.*;
import opennlp.tools.util.eval.FMeasure;

import java.io.*;
import java.nio.charset.Charset;

public class Main {
    public static void main(String[] args) {
        POSModel model = null;

        InputStream dataIn = null;
        try {
            Charset charset = Charset.forName("UTF-8");
            InputStreamFactory isf = new MarkableFileInputStreamFactory(new File("C:\\Users\\CX70\\Documents\\opennlp-rus\\demofile.txt"));
            InputStreamFactory isf2 = new MarkableFileInputStreamFactory(new File("C:\\Users\\CX70\\Documents\\opennlp-rus\\demofile-test.txt"));


            TrainingParameters mlParams = new TrainingParameters();
            mlParams.put("TrainerType", "Event");
            mlParams.put("Cutoff", 5);
            String[] algo = {"MAXENT", "PERCEPTRON", "PERCEPTRON_SEQUENCE"};
            Integer[] iters = {100, 80, 50};

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
                    FMeasure measure = new FMeasure();
                    ObjectStream<String> lineStream2 = new PlainTextByLineStream(isf2, "UTF-8");
                    ObjectStream<POSSample> sampleStream2 = new WordTagSampleStream(lineStream2);
                    while((sample = sampleStream2.read()) != null) {
                        String[] pred = tagger.tag(sample.getSentence());
                        measure.updateScores(pred, sample.getTags());
                    }
                    System.out.println(String.format("alg %s iter %s fm %s prec %s recall %s", alg, it, measure.getFMeasure(), measure.getPrecisionScore(), measure.getRecallScore()));
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
