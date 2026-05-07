idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES bt nvs_flash esp_timer driver)

target_include_directories(__idf_main PRIVATE
    $ENV{IDF_PATH}/components/bt/host/nimble/nimble/porting/npl/freertos/include
    $ENV{IDF_PATH}/components/bt/host/nimble/nimble/porting/nimble/include
    $ENV{IDF_PATH}/components/bt/porting/npl/freertos/include
    $ENV{IDF_PATH}/components/bt/host/nimble/nimble/nimble/include
    $ENV{IDF_PATH}/components/bt/host/nimble/port/include
)