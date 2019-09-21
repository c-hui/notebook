import org.springframework.jdbc.core.support.JdbcDaoSupport;

public class MySql extends JdbcDaoSupport {

    public void execute(String sql)
    {
        getJdbcTemplate().execute(sql);
    }
}
