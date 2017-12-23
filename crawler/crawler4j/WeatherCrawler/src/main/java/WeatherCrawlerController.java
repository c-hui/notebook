import edu.uci.ics.crawler4j.crawler.CrawlConfig;
import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.fetcher.PageFetcher;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtConfig;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtServer;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;


public class WeatherCrawlerController {
    public static void main(String[] args) throws Exception {

        /*
         * crawlStorageFolder is a folder where intermediate crawl data is
         * stored.
         */
        String crawlStorageFolder = "/data/crawl/root";

        /*
         * numberOfCrawlers shows the number of concurrent threads that should
         * be initiated for crawling.
         */
        int numberOfCrawlers = 7;

        CrawlConfig config = new CrawlConfig();
        config.setCrawlStorageFolder(crawlStorageFolder);

        config.setConnectionTimeout(100000);
        config.setSocketTimeout(200000);


        /*
         * Instantiate the controller for this crawl.
         */
        PageFetcher pageFetcher = new PageFetcher(config);
        RobotstxtConfig robotstxtConfig = new RobotstxtConfig();
        robotstxtConfig.setEnabled(false);
        RobotstxtServer robotstxtServer = new RobotstxtServer(robotstxtConfig, pageFetcher);
        CrawlController controller = new CrawlController(config, pageFetcher, robotstxtServer);

        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date date1 = simpleDateFormat.parse("2017-11-01");
        Date date2 = simpleDateFormat.parse("2017-12-01");
        Calendar cal1 = Calendar.getInstance();
        Calendar cal2 = Calendar.getInstance();
        cal1.setTime(date1);
        cal2.setTime(date2);
        Long days = (cal2.getTimeInMillis()-cal1.getTimeInMillis())/1000/3600/24 + 1;
        Long counts = days/5 + 1;
        for (int i=0;i<counts;i++) {
            int y1 = cal1.get(Calendar.YEAR);
            int m1 = cal1.get(Calendar.MONTH) + 1;
            int d1 = cal1.get(Calendar.DAY_OF_MONTH);
            int y2, m2, d2;
            if (i != counts-1) {
                cal1.add(Calendar.DAY_OF_MONTH, 4);
                y2 = cal1.get(Calendar.YEAR);
                m2 = cal1.get(Calendar.MONTH) + 1;
                d2 = cal1.get(Calendar.DAY_OF_MONTH);
                cal1.add(Calendar.DAY_OF_MONTH, 1);
            }
            else {
                y2 = cal2.get(Calendar.YEAR);
                m2 = cal2.get(Calendar.MONTH) + 1;
                d2 = cal2.get(Calendar.DAY_OF_MONTH);
            }
            String url = String.format("http://www.meteomanz.com/" +
                    "sy1?cou=2250&l=1&ty=hp&ind=00000&d1=%02d&m1=%02d&y1=%d&d2=%02d&m2=%02d&y2=%d",
                    d1, m1, y1, d2, m2, y2);

            /*
             * For each crawl, you need to add some seed urls. These are the first
             * URLs that are fetched and then the crawler starts following links
             * which are found in these pages
             */
            controller.addSeed(url);
        }

        /*
         * Start the crawl. This is a blocking operation, meaning that your code
         * will reach the line after this only when crawling is finished.
         */
        controller.start(WeatherCrawler.class, numberOfCrawlers);
    }
}
