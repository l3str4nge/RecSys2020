package common.core.utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class MLFileUtils {

    public static int getNumberOfLines(final String file) throws IOException {
        try (LineNumberReader lnr = new LineNumberReader(
                new FileReader(new File(file)))) {
			lnr.skip(Long.MAX_VALUE);
            return lnr.getLineNumber()+1;
        }
    }

    public static String[] readAllLines(String file) throws IOException {

        List<String> raw = Files.readAllLines(Paths.get(file));
        String[] strs = new String[raw.size()];
        raw.toArray(strs);
        return strs;
    }

}
