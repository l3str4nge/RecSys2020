package recsys2020;

import common.core.utils.MLTimer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Set;

import static java.nio.file.StandardOpenOption.READ;

public class RecSys20Embedding {

    public static final int EMBED_SIZE_FLOAT = 512;
    public static final int EMBED_SIZE_BYTE = EMBED_SIZE_FLOAT * 4;

    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20Embedding");
        timer.tic();
    }

    public long[] tweetOffsets;

    public FileChannel fileChannel;
    public RecSys20FeatExtractor featExtractor;

    public RecSys20Embedding(final RecSys20FeatExtractor featExtractorP) throws Exception {
        this.featExtractor = featExtractorP;

        String embedPath = "/home/layer6/projects/recsys2020/";
//        embedPath = "/media/mvolkovs/external1TB/Data/RecSys2020/Models
//        /Kevin/";

        this.loadOffsets(embedPath);
        this.fileChannel = FileChannel.open(
                Paths.get(embedPath + "embed.out"),
                READ);
        this.test();
    }

    public void loadOffsets(final String embedPath) throws Exception {
        try (BufferedReader reader =
                     new BufferedReader(new FileReader(embedPath + "ids.out"))) {
            long curOffset = 0;
            this.tweetOffsets =
                    new long[this.featExtractor.data.tweetToIndex.size()];
            Arrays.fill(this.tweetOffsets, -1);
            String tweetId;
            while ((tweetId = reader.readLine()) != null) {
                int tweetIndex =
                        this.featExtractor.data.tweetToIndex.get(tweetId);
                this.tweetOffsets[tweetIndex] = curOffset;
                curOffset += EMBED_SIZE_BYTE;
            }
            timer.toc("loadOffsets done nLoaded:" + (curOffset / ((long) EMBED_SIZE_BYTE))
                    + "  nTotal:" + this.featExtractor.data.tweetToIndex.size());
        }
    }

    public static void normalise(final float[] embed) {
        double norm = 0;
        for (int i = 0; i < embed.length; i++) {
            norm = norm + embed[i] * embed[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < embed.length; i++) {
            embed[i] = (float) (embed[i] / norm);
        }
    }

    public float userTweetScore(final int targetUserIndex,
                                final int targetTweetIndex) {
        try {
            if (this.tweetOffsets[targetTweetIndex] < 0) {
                return 0f;
            }

            Set<Integer>[] userEngage =
                    this.featExtractor.userToEngage[targetUserIndex];
            double[] targetUserEmbed = new double[EMBED_SIZE_FLOAT];
            int count = 0;
            for (int i = 0; i < RecSys20FeatExtractor.N_ENGAGE; i++) {
                if (userEngage[i] == null) {
                    continue;
                }

                for (int engageIndex : userEngage[i]) {
                    int[] engage = this.featExtractor.data.engage[engageIndex];
                    int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                    if (this.tweetOffsets[tweetIndex] < 0) {
                        continue;
                    }
                    count++;
                    float[] tweetEmbed = getTweetEmbed(
                            tweetIndex,
                            this.tweetOffsets,
                            this.fileChannel);
                    normalise(tweetEmbed);

                    for (int j = 0; j < tweetEmbed.length; j++) {
                        targetUserEmbed[j] = targetUserEmbed[j] + tweetEmbed[j];
                    }
                }
            }
            if (count == 0) {
                return 0f;
            }

            float[] targetTweetEmbed = getTweetEmbed(
                    targetTweetIndex,
                    this.tweetOffsets,
                    this.fileChannel);
            normalise(targetTweetEmbed);

            double score = 0;
            for (int i = 0; i < targetUserEmbed.length; i++) {
                score += (targetUserEmbed[i] / count) * ((double) targetTweetEmbed[i]);
            }
            return (float) score;

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e.getMessage());
        }
    }

    public float tweetTweetScore(final int targetTweetIndex1,
                                 final int targetTweetIndex2,
                                 final boolean normalise) {
        try {
            if (this.tweetOffsets[targetTweetIndex1] < 0 || this.tweetOffsets[targetTweetIndex2] < 0) {
                return 0f;
            }

            float[] targetTweetEmbed1 = getTweetEmbed(
                    targetTweetIndex1,
                    this.tweetOffsets,
                    this.fileChannel);
            if (normalise == true) {
                normalise(targetTweetEmbed1);
            }

            float[] targetTweetEmbed2 = getTweetEmbed(
                    targetTweetIndex2,
                    this.tweetOffsets,
                    this.fileChannel);
            if (normalise == true) {
                normalise(targetTweetEmbed2);
            }

            double score = 0;
            for (int i = 0; i < targetTweetEmbed1.length; i++) {
                score += ((double) targetTweetEmbed1[i]) * ((double) targetTweetEmbed2[i]);
            }
            return (float) score;

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e.getMessage());
        }
    }

    public void compare(final String id,
                        final float[] firstFive) throws Exception {
        float[] tweetEmbed = getTweetEmbed(
                this.featExtractor.data.tweetToIndex.get(id),
                this.tweetOffsets,
                this.fileChannel);
        for (int i = 0; i < firstFive.length; i++) {
            float diff = Math.abs(firstFive[i] - tweetEmbed[i]);
            if (diff > 1e-4f) {
                throw new IllegalStateException(id + " failed");
            }
        }
    }

    public void test() throws Exception {

        String id = "E459F4756641E2AE53D99837D1ADF23A";
        float[] firstFive = new float[]{0.04577632f, 0.00812454f, 0.01727203f
                , -0.02523926f, -0.10241052f};
        this.compare(id, firstFive);

        id = "7E3ABE5CB91D61E40EC22F74F6BD960A";
        firstFive = new float[]{-0.0952917f, -0.02707538f, 0.01605354f,
                -0.04456393f, -0.04462935f};
        this.compare(id, firstFive);

        id = "7FC54E66347A4EDC2965E895F8BE0E14";
        firstFive = new float[]{0.03145852f, -0.04131042f, 0.0215825f,
                0.00057718f, 0.01727196f};
        this.compare(id, firstFive);

        id = "27FD1CC658E4277316824130AB398A84";
        firstFive = new float[]{-0.00900629f, -0.06874217f, -0.05926217f,
                -0.05144354f, -0.06257589f};
        this.compare(id, firstFive);

        id = "3D8671C5AD9AC8D6F5ACED4EB80FDA2E";
        firstFive = new float[]{0.01889422f, -0.08710929f, 0.01041983f,
                0.01482615f, 0.03022419f};
        this.compare(id, firstFive);

        id = "7DF1738B6F6CF2900AA68BB952CDAAF6";
        firstFive = new float[]{-0.00934002f, 0.00059972f, -0.02025026f,
                -0.03694036f, -0.09379258f};
        this.compare(id, firstFive);

        id = "B8A8C04A53C6F71F9F4EBF322CD9A5AF";
        firstFive = new float[]{-0.03128658f, 0.05725005f, 0.02025172f,
                0.00197596f, -0.10150675f};
        this.compare(id, firstFive);

        id = "AB8BA813CB74FF87635A6E4C1B8E93FF";
        firstFive = new float[]{0.02838854f, -0.0213445f, 0.03007559f,
                0.02464638f, -0.10947238f};
        this.compare(id, firstFive);

        id = "2977EDD9F928534ACCA3245D0B2CC303";
        firstFive = new float[]{-0.07248894f, -0.0489621f, 0.04741488f,
                0.05545109f, 0.06373998f};
        this.compare(id, firstFive);

        id = "11BCE4EFEB24F4A907675CD33A5118F1";
        firstFive = new float[]{0.01371974f, -0.05610174f, -0.01493494f,
                0.01020032f, -0.10853902f};
        this.compare(id, firstFive);

        id = "7664B1B15BEADDB9A326B4F1D482E61A";
        firstFive = new float[]{-0.02038005f, -0.0206392f, 0.00915268f,
                -0.00343527f, 0.04822627f};
        this.compare(id, firstFive);

        timer.toc("test passed");
    }

    public static float[] getTweetEmbed(final int tweetIndex,
                                        final long[] tweetOffsets,
                                        final RandomAccessFile randomAccess) throws Exception {
        byte[] embed = new byte[EMBED_SIZE_BYTE];
        synchronized (randomAccess) {
            randomAccess.seek(tweetOffsets[tweetIndex]);
            int nRead = randomAccess.read(embed);
            if (nRead != EMBED_SIZE_BYTE) {
                throw new IllegalStateException("nRead != EMBED_SIZE_BYTE");
            }
        }
        return toFloatArray(embed);
    }

    public static float[] getTweetEmbed(final int tweetIndex,
                                        final long[] tweetOffsets,
                                        final FileChannel fileChannel) throws Exception {

        ByteBuffer buff = ByteBuffer.allocate(EMBED_SIZE_BYTE);
        int nRead = fileChannel.read(buff, tweetOffsets[tweetIndex]);
        if (nRead != EMBED_SIZE_BYTE) {
            throw new IllegalStateException("nRead != EMBED_SIZE_BYTE");
        }
        return toFloatArray(buff.array());
    }

    public static float[] toFloatArray(byte[] bytes) {
        FloatBuffer floatBuffer = ByteBuffer.wrap(bytes).asFloatBuffer();

        float[] floatArray = new float[EMBED_SIZE_FLOAT];
        floatBuffer.get(floatArray);

        return floatArray;
    }

    public static void main(final String[] args) {
        String bertPath = "/home/layer6/projects/recsys2020/";
        try (FileChannel fileChannel = FileChannel.open(Paths.get(bertPath +
                "embed.out"), READ)) {
            float[] tweetEmbed = RecSys20Embedding.getTweetEmbed(
                    0,
                    new long[]{0L},
                    fileChannel);
            System.out.println(tweetEmbed.length);
            System.out.println(Arrays.toString(tweetEmbed));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


//cat vector_chunk_0.bin vector_chunk_1.bin vector_chunk_2.bin \
//vector_chunk_3.bin vector_chunk_4.bin vector_chunk_5.bin \
//vector_chunk_6.bin vector_chunk_7.bin \
//vector_chunk_8.bin vector_chunk_9.bin \
//vector_chunk_10.bin  > /home/layer6/projects/recsys2020/embed.out

//cat id_chunk0.csv id_chunk1.csv id_chunk2.csv id_chunk3.csv id_chunk4.csv \
//id_chunk5.csv id_chunk6.csv id_chunk7.csv id_chunk8.csv id_chunk9.csv \
//id_chunk10.csv > /home/layer6/projects/recsys2020/ids.out