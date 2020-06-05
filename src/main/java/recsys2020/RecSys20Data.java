package recsys2020;

import common.core.feature.MLSparseFeature;

import java.io.Serializable;
import java.util.Map;

public class RecSys20Data implements Serializable {

    public static final long serialVersionUID = 1;

    public enum TweetFeature {
        text_tokens(0),
        hashtags(1),
        //        tweet_id,
        present_media(3),
        present_links(4),
        present_domains(5),
        tweet_type(6),
        language(7);
        //        timestamp

        public int index;

        TweetFeature(final int indexP) {
            this.index = indexP;
        }
    }

    public enum EngageType {
        REPLY(0),
        RETWEET(1),
        COMMENT(2),
        LIKE(3);

        // index in the long[][] engageAction array
        public int index;

        public static EngageType fromString(final String string) {
            for (EngageType action : EngageType.values()) {
                if (string.equals(action.toString()) == true) {
                    return action;
                }
            }
            throw new IllegalArgumentException("unsupported EngageType: " + string);
        }

        EngageType(final int indexP) {
            this.index = indexP;
        }
    }


    public Map<String, Integer> userToIndex;
    public long[] userCreation; //account creation time


    public Map<String, Integer> tweetToIndex;
    public long[] tweetCreation; //tweet creation time
    public Map<TweetFeature, MLSparseFeature> tweetFeatures;

    //FORMAT:
    //tweetIndex,
    //creatorIndex,
    //creatorFollowers,
    //creatorFollowing,
    //creatorVerified,
    //userIndex,
    //useFollowers,
    //userFollowing,
    //userVerified,
    //follow
    public int[][] engage;

    //FORMAT:
    //replyTime,
    //retweetTime,
    //commentTime,
    //likeTime
    public long[][] engageAction;

    public int lbEngageIndex;
    public int testEngageIndex;
    public int testEngageIndexEnd;

}
