package mainCpp;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class MainCPP extends Application{

    @Override
    public void start(Stage primaryStage) throws Exception {
        try {
            FXMLLoader root = new FXMLLoader(getClass().getResource("sample.fxml"));
            BorderPane rootElement = root.load();

            primaryStage.setTitle("Hello World");
            primaryStage.setScene(new Scene(rootElement, 1200, 800));
            primaryStage.setResizable(false);
            primaryStage.show();

            ControllerCPP controller = root.getController();
            primaryStage.setOnCloseRequest((we -> controller.setClosed()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
