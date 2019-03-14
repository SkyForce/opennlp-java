package csc;

import opennlp.tools.postag.*;
import opennlp.tools.util.*;

import java.io.*;
import java.nio.charset.Charset;

public class Main {
    public static void main(String[] args) {
        POSModel model = null;

        InputStream dataIn = null;
        try {
            Charset charset = Charset.forName("UTF-8");
            InputStreamFactory isf = new MarkableFileInputStreamFactory(new File("C:\\Users\\CX70\\Documents\\opennlp-rus\\demofile.txt"));
            ObjectStream<String> lineStream = new PlainTextByLineStream(isf, "UTF-8");
            ObjectStream<POSSample> sampleStream = new WordTagSampleStream(lineStream);

            model = POSTaggerME.train("ru", sampleStream, TrainingParameters.defaultParams(), new POSTaggerFactory());
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
