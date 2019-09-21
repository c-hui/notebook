import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.util.regex.Pattern;

public class WeatherCrawler extends WebCrawler {

    private static final Pattern Link = Pattern.compile("http://www.meteomanz.com/.*&np=\\d+&.*");
    private static final Pattern NoLink = Pattern.compile("http://www.meteomanz.com/.*&np=1&.*");
    private static MySql mysql;


    public WeatherCrawler() {
        mysql = (MySql) OnlyApplicateContext.getInstance().getContext().getBean("mysqlTest");
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
        if (page.getParseData() instanceof HtmlParseData) {
            HtmlParseData htmlParseData = (HtmlParseData) page.getParseData();
            String html = htmlParseData.getHtml();
            Document doc = Jsoup.parse(html);

            Elements trs = doc.select("tr");
            trs.remove(0);
            for (Element tr : trs) {
                StringBuilder sb = new StringBuilder();
                sb.append("insert into weather_history(Station, Date, Time, Temp, RelHum," +
                        " Pressure, WindDir, WinsSpeed, Clouds, LowClouds, MediumClouds," +
                        " HighClouds, Prec, MaxTemp, MinTemp, Conditions) values (");
                Elements tds = tr.select("td");
                int i = 0;
                for (Element td : tds) {
                    i += 1;
                    String text = td.text();
                    if (text.equals("") || text.equals("N/A") || text.equals("-"))
                        sb.append("NULL");
                    else sb.append("\"").append(text).append("\"");
                    if (i < 16) sb.append(",");
                }
                sb.append(")");
                mysql.execute(sb.toString());
            }
        }
    }
}
