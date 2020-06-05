package recsys2020;

import common.core.linalg.FloatElement;

import recsys2020.RecSys20Data.EngageType;

public class RecSys20Eval {

    public double logL;
    public double prAUC;


    public void evaluate(final EngageType engage,
                         final FloatElement[] preds,
                         final RecSys20Data data) {
        double ctr = 0;
        for (int i = 0; i < preds.length; i++) {
            //compute data ctr
            FloatElement element = preds[i];
            long[] engageAction = data.engageAction[element.getIndex()];
            if (engageAction != null && engageAction[engage.index] > 0) {
                ctr++;
            }
        }
        ctr = ctr / preds.length;

        this.logL = 0;
        this.prAUC = 0;

        double logLCTR = 0;
        double[] precision = new double[preds.length];
        double[] recall = new double[preds.length];
        double nRelevant = 0;
        for (int i = 0; i < preds.length; i++) {
            FloatElement element = preds[i];
            long[] engageAction = data.engageAction[element.getIndex()];

            //get ground truth for this prediction
            int target = 0;
            if (engageAction != null && engageAction[engage.index] > 0) {
                target = 1;
                nRelevant++;
            }

            //compute logL
            if (target == 1) {
                this.logL -= Math.log(element.getValue() + 1e-10);
                logLCTR -= Math.log(ctr);
            } else {
                this.logL -= Math.log(1 - element.getValue() + 1e-10);
                logLCTR -= Math.log(1 - ctr);
            }

            //precision and recall
            precision[i] = nRelevant / (i + 1);
            recall[i] = nRelevant;
        }

        this.logL = logL / preds.length;
        logLCTR = logLCTR / preds.length;

        this.logL = (1.0 - logL / logLCTR) * 100.0;
        System.out.println(String.format("nEval:%d  engage:%s  logL:%.4f",
                preds.length, engage, this.logL));


        for (int i = 0; i < recall.length; i++) {
            recall[i] = recall[i] / nRelevant;
            if (i > 0) {
                this.prAUC += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2;
            }
        }
        System.out.println(String.format("nEval:%d  engage:%s  prAUC:%.4f",
                preds.length, engage, this.prAUC));
    }
}