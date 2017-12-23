import org.springframework.context.support.ClassPathXmlApplicationContext;
public class OnlyApplicateContext {
    static OnlyApplicateContext instance = null;

    private ClassPathXmlApplicationContext context = null;

    public OnlyApplicateContext(){
        context = new ClassPathXmlApplicationContext("classpath:spring.xml");
    }

    public static OnlyApplicateContext getInstance(){
        if(instance==null){
            synchronized (OnlyApplicateContext.class)  {
                if(instance==null) instance = new OnlyApplicateContext();
            }
            System.out.println("old context");
        }
        else{
            System.out.println("new context");
        }

        return instance;
    }

    public ClassPathXmlApplicationContext getContext()   {
        return context;
    }

}
