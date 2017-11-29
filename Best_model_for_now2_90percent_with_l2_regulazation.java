package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb;

/**
 * Animal Classification
 *
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 *
 * References:
 *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 *
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 *  - Add additional images to the dataset
 *  - Apply more transforms to dataset
 *  - Increase epochs
 *  - Try different model configurations
 *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class Best_model_for_now2_90percent_with_l2_regulazation {
    protected static final Logger log = LoggerFactory.getLogger(Best_model_for_now2_90percent_with_l2_regulazation.class);
    protected static int height = 64;
    protected static int width = 64;
    protected static int channels = 3;
    protected static int numExamples = 4000;
    protected static int numLabels = 4;
    protected static int batchSize = 25;

    protected static long seed = 421;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 32;
    protected static double splitTrainTest = 0.9;
    protected static boolean save = false;

    protected static String modelType = "LeNet"; // LeNet, AlexNet or Custom but you need to fill it out

    public void run(String[] args) throws Exception {

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        System.out.println(System.getProperty("user.dir") + "======" + "dl4j-examples/src/main/resources/animals/");
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
       ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform flipTransform3 = new FlipImageTransform(new Random(456));
        ImageTransform flipTransform4 = new FlipImageTransform(new Random(789));
        ImageTransform flipTransform5 = new FlipImageTransform(new Random(978));
        ImageTransform flipTransform6 = new FlipImageTransform(new Random(885));
        ImageTransform flipTransform7 = new FlipImageTransform(new Random(887));
        ImageTransform flipTransform8 = new FlipImageTransform(new Random(886));
        ImageTransform flipTransform9 = new FlipImageTransform(new Random(889));


        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        ImageTransform warpTransform2 = new WarpImageTransform(rng, 43);
        ImageTransform warpTransform3= new WarpImageTransform(rng, 44);
        ImageTransform warpTransform4 = new WarpImageTransform(rng, 55);
        ImageTransform warpTransform5= new WarpImageTransform(rng, 66);
        ImageTransform warpTransform6 = new WarpImageTransform(rng, 77);


      //  ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
     //   ImageTransform colorTransform2 = new ColorConversionTransform(new Random(45), COLOR_BGR2YCrCb);
     //   ImageTransform colorTransform3 = new ColorConversionTransform(new Random(39), COLOR_BGR2YCrCb);
    //    ImageTransform colorTransform4 = new ColorConversionTransform(new Random(47), COLOR_BGR2YCrCb);
     //   ImageTransform colorTransform5 = new ColorConversionTransform(new Random(256), COLOR_BGR2YCrCb);



        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1});
       // List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{warpTransform6,flipTransform1, flipTransform2,warpTransform,flipTransform3,warpTransform3,flipTransform4,warpTransform2,flipTransform5,warpTransform5,flipTransform6,warpTransform5,flipTransform7,flipTransform8,flipTransform9,warpTransform6});

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
//        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();





        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = lenetModel();
                break;

            case "custom":
                network = customModel();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        network.init();
       // network.setListeners(new ScoreIterationListener(listenerFreq));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners((IterationListener)new StatsListener( statsStorage),new ScoreIterationListener(iterations));
        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        //ui


        log.info("Train model....");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainIter);

        // Train with transformations



/*       for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter);
            network.fit(trainIter);
        }*/




        log.info("Evaluate model.... on training set");
        recordReader.initialize(trainData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));


        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }
        log.info("****************Example finished********************");
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv7x7(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv9x9(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv8x8(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv4x4(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }


    private ConvolutionLayer conv2x2(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.003) // tried 0.0001, 0.0005
            .activation(Activation.RELU)
            .learningRate(0.005) // tried 0.00001, 0.00005, 0.000001
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.9))
            //         .updater(new AdaDelta())
            .list()
            .layer(0, convInit("cnn1", channels, 64 ,  new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}, 0))
            .layer(1, maxPool("maxpool1", new int[]{2,2}))
            .layer(2, conv2x2("cnn4*2*2", 128,  0))
            .layer(3, maxPool("maxool4", new int[]{2,2}))
            .layer(4, conv3x3("cnn33*3", 64, 0))
            .layer(5, maxPool("maxool3", new int[]{2,2}))
            //      .layer(6,new DropoutLayer.Builder().name("dropout1").dropOut(0.25).build())
            .layer(6, new DenseLayer.Builder().nOut(500).build())
            //  .layer(8,new DropoutLayer.Builder().name("dropout1").dropOut(0.25).build())
            .layer(7, new DenseLayer.Builder().nOut(500).build())
            .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true).pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);



    }



    public  MultiLayerNetwork customModel() {
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed) //
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-1)
            .learningRateScoreBasedDecayRate(1e-1)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .list()
            .layer(0, convInit("cnn1", channels, 64, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(1, maxPool("maxpool1", new int[]{2,2}))
            .layer(2, convInit("cnn2", 64, 128, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(3, maxPool("maxpool2", new int[]{2,2}))
            .layer(4, convInit("cnn3", 128, 256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(5, convInit("cnn4", 256, 256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(6, maxPool("maxpool3", new int[]{2,2}))
            .layer(7, convInit("cnn5", 256, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(8, convInit("cnn6", 512, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(9, maxPool("maxpool4", new int[]{2,2}))
            .layer(10, convInit("cnn7", 512, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(11, convInit("cnn8", 512, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
            .layer(12, maxPool("maxpool5", new int[]{2,2}))
            .layer(13, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(14, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);
    }

    public static void main(String[] args) throws Exception {
        new Best_model_for_now2_90percent_with_l2_regulazation().run(args);
    }

}
