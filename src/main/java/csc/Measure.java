package csc;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class Measure {
    HashMap<String, Integer> dictPred = new HashMap<String, Integer>();
    HashMap<String, Integer> dictTarget = new HashMap<String, Integer>();
    HashMap<String, Integer> dictEqual = new HashMap<String, Integer>();
    public void update(Object[] pred, Object[] target) {
        for(int i = 0; i < pred.length; i++) {
            if(pred[i].equals(target[i])) {
                dictEqual.put(pred[i].toString(), dictEqual.getOrDefault(pred[i], 0) + 1);
            }
            dictPred.put(pred[i].toString(), dictPred.getOrDefault(pred[i], 0) + 1);
            dictTarget.put(target[i].toString(), dictTarget.getOrDefault(target[i], 0) + 1);
        }
    }

    public double getPrecision() {
        Set<String> keys = new HashSet(dictPred.keySet());
        keys.addAll(dictTarget.keySet());
        double prec = 0;
        for(String key : keys) {
            prec += (double) dictEqual.getOrDefault(key, 0) / dictPred.getOrDefault(key, 1);
        }
        return prec / keys.size();
    }

    public double getRecall() {
        Set<String> keys = new HashSet(dictPred.keySet());
        keys.addAll(dictTarget.keySet());
        double rec = 0;
        for(String key : keys) {
            rec += (double) dictEqual.getOrDefault(key, 0) / dictTarget.getOrDefault(key, 1);
        }
        return rec / keys.size();
    }

    public double getF1() {
        return 2 * getPrecision() * getRecall() / (getPrecision() + getRecall());
    }

    public double getAccuracy() {
        return dictPred.values().stream().mapToDouble(i -> i.doubleValue()).sum() / dictEqual.values().stream().mapToInt(i -> i.intValue()).sum();
    }
}
