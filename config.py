TEST_CITY = "Shanghai"
VALIDATION_DATE = "2015-01-01"
PATTERN_FOR_TARGET = "PM"
TARGET_VARIABLE = "PM"
VARIABLES_CONSIDERED = ["date", "city", "PM", "HUMI", "PRES", "TEMP", "Iws", "precipitation", "Iprec"]
NUMERICAL_VARIABLES = ["HUMI", "PRES", "TEMP", "Iws", "precipitation", "Iprec"]
CITIES = ["Beijing", "Chengdu", "Guangzhou", "Shanghai", "Shenyang"]


EPOCHS = 20
BATCH_SIZE = 12
LEARNING_RATE = 0.01
NUM_HIDDEN_UNITS = 16
NUM_LAYERS = 2
DROPOUT = 0.1
SEQUENCE_LENGTH = 6
