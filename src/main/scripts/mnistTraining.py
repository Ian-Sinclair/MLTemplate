from src.main.classes.dataHandlers import MnistDataHandler
from src.main.classes.modelArchitectures.CNN import CNN

if __name__ == "__main__": 

    RESULTS_PATH_BASE = "models/MNIST/WGAN_GP/"    
    SAVED_MODEL_PATH = RESULTS_PATH_BASE + "saved_models/"
    ARCHITECTURE_DIAGRAM_PATH = RESULTS_PATH_BASE + "visualizations/architecture/"
    METRICS_SAVE_PATH = RESULTS_PATH_BASE + "visualizations/results/"

    data_hanlder = MnistDataHandler().collect_data().process_data()
    (x_train, y_train), (x_test, y_test) = data_hanlder.get_processed_data()

    cnn = CNN(input_size=(28,28,1),
              filter_sizes=[32,64,64],
              kernel_sizes=[3,3,3],
              strides=[1,2,1],
              output_dim=MnistDataHandler.NUM_LABELS)
    
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    EPOCHS = 5

    cnn.save_architecture_diagram(path=ARCHITECTURE_DIAGRAM_PATH,file_name="WGAN_GP_ARCHITECTURE")
    cnn = cnn.save_model(SAVED_MODEL_PATH)

    cnn.train(
        x_train=x_train,
        y_train=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        save_folder=METRICS_SAVE_PATH
    )
    
    cnn = cnn.save_model(SAVED_MODEL_PATH).load_model(SAVED_MODEL_PATH)

    print(f"Process Complete")