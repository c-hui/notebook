import com.csvreader.CsvWriter;
import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Pattern;

public class WeatherCrawler extends WebCrawler {

    private static final Pattern Link = Pattern.compile("http://www.meteomanz.com/.*&np=\\d+&.*");
    private static final Pattern NoLink = Pattern.compile("http://www.meteomanz.com/.*&np=1&.*");

    /** data save path */
    private final static String CSV_PATH = "data/crawl/weather.csv";

    private final File csv;
    private CsvWriter cw;

    public WeatherCrawler() throws IOException {
        csv = new File(CSV_PATH);
        if (csv.isFile()) {
            csv.delete();
        }
        cw = new CsvWriter(new FileWriter(csv, true), ',');
        cw.write("Station");
        cw.write("Date");
        cw.write("UTC time");
        cw.write("Temp.(ºC)");
        cw.write("Rel. Hum.(%)");
        cw.write("Pressure/Geopot.");
        cw.write("Wind dir");
        cw.write("Wins speed(Km/h)");
        cw.write("Clouds");
        cw.write("Low clouds");
        cw.write("Medium clouds");
        cw.write("High clouds");
        cw.write("Prec.(mm)");
        cw.write("Max temp.(ºC)");
        cw.write("Min temp.(ºC)");
        cw.write("Conditions");
        cw.endRecord();
        cw.close();
    }


    /**
     * You should implement this function to specify whether the given url
     * should be crawled or not (based on your crawling logic).
     */
    @Override
    public boolean shouldVisit(Page referringPage, WebURL url) {
        String urlString = url.getURL();
        if (Link.matcher(urlString).matches())
            if (!NoLink.matcher(urlString).matches())
                return true;
        return false;
    }

    /**
     * This function is called when a page is fetched and ready to be processed
     * by your program.
     */
    @Override
    public void visit(Page page) {
        String url = page.getWebURL().getURL();
        System.out.println(url);
        HtmlParseData htmlParseData = (HtmlParseData) page.getParseData();
        String html = htmlParseData.getHtml();
        Document doc = Jsoup.parse(html);
        try {
            cw = new CsvWriter(new FileWriter(csv, true), ',');
            Elements trs = doc.select("tr");
            trs.remove(0);
            for (Element tr : trs) {
                Elements tds = tr.select("td");
                for (Element td : tds) {
                    String text = td.text();
                    if (text.equals("") || text.equals("N/A"))
                        cw.write("-");
                    else cw.write(text);
                }
                cw.endRecord();
                cw.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            cw.close();
        }
    }
}
