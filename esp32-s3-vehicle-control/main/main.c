#include <stdio.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include <driver/ledc.h>
#include "driver/gpio.h"  
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// For bluetooth set up 
#include "nimble/nimble_port_freertos.h"
#include "nimble/nimble_port.h"
#include "nvs_flash.h"
#include "host/ble_hs.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define SERVICE_UUID        0xABCD
#define CHAR_UUID           0x1234

#define INPUT_TIMEOUT_MS 200

#define MOTOR_IN1_PIN           GPIO_NUM_15   // IN1 pin connected to motor driver (left)
#define MOTOR_IN2_PIN          GPIO_NUM_7  // IN2 pin connected to motor driver (left)
#define MOTOR_ENABLE_PIN_LEFT  GPIO_NUM_8  // Enable pin for PWM control on left motor

#define MOTOR_IN3_PIN          GPIO_NUM_6  // IN3 pin connected to motor driver (right)
#define MOTOR_IN4_PIN          GPIO_NUM_5  // IN4 pin connected to motor driver (right)
#define MOTOR_ENABLE_PIN_RIGHT GPIO_NUM_4  // Same as above, for right motor 


#define MOTOR_PWM_FREQ          5000  // Frequency in Hz for PWM
#define MOTOR_PWM_CHANNEL_RIGHT LEDC_CHANNEL_0
#define MOTOR_PWM_CHANNEL_LEFT  LEDC_CHANNEL_1
#define MOTOR_PWM_MODE          LEDC_LOW_SPEED_MODE
#define MOTOR_PWM_TIMER         LEDC_TIMER_0
#define MOTOR_PWM_RES           LEDC_TIMER_10_BIT  // PWM resolution (10-bit)
#define MAX_DUTY_CYCLE          1023  // Maximum duty cycle for 10-bit resolution (1023)

void start_advertising(void);
static int gap_event_handler(struct ble_gap_event *event, void *arg);

// Speed control
int duty_cycle = 700;  //1023 is max it can go, 
                       // after testing it seems that 700 is the lowest it can go while still functioning normally

// Tracks the last time a command was received via BLE
static volatile int64_t last_cmd_time = 0;

void stop() {
    // set and apply via update. Set pwn to 0 for either stop or resetting purposes
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, 0);  
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, 0);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
}

void enable_motors() {
    // set and apply via update
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, duty_cycle);
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, duty_cycle);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
}

void backward() {
    // Set spin diretion to backward
    stop();
    vTaskDelay(pdMS_TO_TICKS(10));  // give driver time to settle
    gpio_set_level(MOTOR_IN3_PIN, 0);
    gpio_set_level(MOTOR_IN4_PIN, 1);
    gpio_set_level(MOTOR_IN1_PIN, 0);
    gpio_set_level(MOTOR_IN2_PIN, 1);
    enable_motors();
} 
void forward() {
    // Set spin diretion to forward
    stop();
    vTaskDelay(pdMS_TO_TICKS(10));  // give driver time to settle
    gpio_set_level(MOTOR_IN3_PIN, 1);
    gpio_set_level(MOTOR_IN4_PIN, 0);
    gpio_set_level(MOTOR_IN1_PIN, 1);
    gpio_set_level(MOTOR_IN2_PIN, 0);
    enable_motors();
}

void left() {
    // Move right wheel only to turn left
    stop(); 
    vTaskDelay(pdMS_TO_TICKS(10));  // give driver time to settle
    gpio_set_level(MOTOR_IN3_PIN, 1);
    gpio_set_level(MOTOR_IN4_PIN, 0);
    gpio_set_level(MOTOR_IN1_PIN, 0);
    gpio_set_level(MOTOR_IN2_PIN, 0);

    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, duty_cycle);  
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, 0);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
}

void right() {
    // Move left wheel only to turn right
    stop();
    vTaskDelay(pdMS_TO_TICKS(10));  // give driver time to settle
    gpio_set_level(MOTOR_IN3_PIN, 0);
    gpio_set_level(MOTOR_IN4_PIN, 0);
    gpio_set_level(MOTOR_IN1_PIN, 1);
    gpio_set_level(MOTOR_IN2_PIN, 0);

    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, 0);  
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, duty_cycle);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
}

// More bluetooth stuff 

static int gap_event_handler(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
        case BLE_GAP_EVENT_DISCONNECT:
            printf("Disconnected, restarting advertising...\n");
            start_advertising();  // restart when client disconnects
            break;
        case BLE_GAP_EVENT_CONNECT:
            printf("Connected!\n");
            break;
    }
    return 0;
}
// Advertise so computer can find it
void start_advertising(void) {
    struct ble_hs_adv_fields fields = {0};
    const char *name = "ESP32-Vehicle";
    fields.name = (uint8_t *)name;
    fields.name_len = strlen(name);
    fields.name_is_complete = 1;
    ble_gap_adv_set_fields(&fields);

    struct ble_gap_adv_params adv_params = {0};
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER, &adv_params, gap_event_handler, NULL);
}

// Called when BLE stack finishes initializing, start advertising here
static void on_sync(void) {
    start_advertising();  // the function from my previous message
    printf("BLE ready, advertising as ESP32-Vehicle\n");
}

//runs the BLE host loop
static void ble_host_task(void *param) {
    nimble_port_run();           // blocks here, handling all BLE events internally
    nimble_port_freertos_deinit();
}

// Watchdog: if no command received in 500ms, stop motors
static void timeout_watchdog(void *arg) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(100));  // check every 100ms
        int64_t now = esp_timer_get_time();
        if (last_cmd_time > 0 && (now - last_cmd_time) > 500000) { 
            stop();
        }
    }
}

// Callback fires when computer write byte to serial
static int motor_char_access(uint16_t conn_handle, uint16_t attr_handle,
                              struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        last_cmd_time = esp_timer_get_time(); 
        uint8_t cmd = ctxt->om->om_data[0];  // byte sent by computer 
        printf("Received command: %c\n", cmd); 
        switch (cmd) {
            case 'w': printf("Going forward\n"); forward();  break;
            case 'a': printf("Going left\n");  left();     break;
            case 's': printf("Going backward\n"); backward(); break;
            case 'd': printf("Going right\n"); right();    break;
            default:  printf("Stopping\n");  stop();     break;
        }
    }
    return 0;
}

// GATT service table
static const struct ble_gatt_svc_def gatt_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID16_DECLARE(SERVICE_UUID),
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = BLE_UUID16_DECLARE(CHAR_UUID),
                .access_cb = motor_char_access,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            { 0 }  // terminator
        },
    },
    { 0 }  // terminator
};


void app_main(void) {
    // Configure GPIO for motor direction control
    esp_rom_gpio_pad_select_gpio(MOTOR_IN1_PIN);
    gpio_set_direction(MOTOR_IN1_PIN, GPIO_MODE_OUTPUT);
    esp_rom_gpio_pad_select_gpio(MOTOR_IN2_PIN);
    gpio_set_direction(MOTOR_IN2_PIN, GPIO_MODE_OUTPUT);

    esp_rom_gpio_pad_select_gpio(MOTOR_IN3_PIN);
    gpio_set_direction(MOTOR_IN3_PIN, GPIO_MODE_OUTPUT);
    esp_rom_gpio_pad_select_gpio(MOTOR_IN4_PIN);
    gpio_set_direction(MOTOR_IN4_PIN, GPIO_MODE_OUTPUT);


    // Configure PWM timer
    ledc_timer_config_t pwm_timer = {
        .speed_mode       = MOTOR_PWM_MODE,
        .duty_resolution  = MOTOR_PWM_RES,
        .timer_num        = MOTOR_PWM_TIMER,
        .freq_hz          = MOTOR_PWM_FREQ,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ledc_timer_config(&pwm_timer);

    // Configure PWM channel right wheel
    ledc_channel_config_t pwm_channel_right = {
        .gpio_num       = MOTOR_ENABLE_PIN_RIGHT,
        .speed_mode     = MOTOR_PWM_MODE,
        .channel        = LEDC_CHANNEL_0,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = MOTOR_PWM_TIMER,
        .duty           = 0,
        .hpoint         = 0
    };
    ledc_channel_config(&pwm_channel_right);

    // Conifgure PWM channel left wheel
    ledc_channel_config_t pwm_channel_left = {
        .gpio_num       = MOTOR_ENABLE_PIN_LEFT,
        .speed_mode     = MOTOR_PWM_MODE,
        .channel        = LEDC_CHANNEL_1,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = MOTOR_PWM_TIMER,
        .duty           = 0,
        .hpoint         = 0
    };
    ledc_channel_config(&pwm_channel_left);

    // give hardware time to stabilize/initialize everything
    vTaskDelay(pdMS_TO_TICKS(500)); 

    // NimBLE (Bluetooth) Initialization 
    nvs_flash_init();                   
    nimble_port_init();

    ble_svc_gap_init();                
    ble_svc_gatt_init();

    ble_gatts_count_cfg(gatt_services); 
    ble_gatts_add_svcs(gatt_services);

    ble_hs_cfg.sync_cb = on_sync;    

    nimble_port_freertos_init(ble_host_task); 

    // Timeout watchdog task
    xTaskCreate(timeout_watchdog, "watchdog", 2048, NULL, 5, NULL);

    // REMEMBER, BLE runs in its own task, not part of main
    while (true) {
        stop();
        vTaskDelay(pdMS_TO_TICKS(1000));

        // Old code before bluetooth, REMEMBER TO DELETE LATER!!
        // stop();
        // char key = getchar(); 

        // // Clear buff of everything received, read most recent
        // int ch;
        // while ((ch = fgetc(stdin)) != EOF);

        // switch (key) {
        //     case 'w': // Forward
        //         forward();
        //         break; 

        //     case 'a': // Left
        //         left();
        //         break; 

        //     case 's': // Backward 
        //         backward();
        //         break; 

        //     case 'd': // Right
        //         right();
        //         break; 

        //     default: 
        //         stop();
        //         break; 
            
        // }      

        // if (key == 'a' || key == 'd') {
        //     // Shorter delay for more controlled turns
        //     vTaskDelay(pdMS_TO_TICKS(200)); 
        // }
        // else {
        //     vTaskDelay(pdMS_TO_TICKS(500)); 
        // }
    }
            

}







// No bluetooth ver

// #include <stdio.h>
// #include <stdlib.h>
// #include "esp_log.h"
// #include "esp_timer.h"
// #include <driver/ledc.h>
// #include "driver/gpio.h"  
// #include "driver/uart.h"
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"

// #define INPUT_TIMEOUT_MS 200

// #define MOTOR_IN1_PIN           GPIO_NUM_15   // IN1 pin connected to motor driver (left)
// #define MOTOR_IN2_PIN          GPIO_NUM_7  // IN2 pin connected to motor driver (left)
// #define MOTOR_ENABLE_PIN_LEFT  GPIO_NUM_8  // Enable pin for PWM control on left motor

// #define MOTOR_IN3_PIN          GPIO_NUM_6  // IN3 pin connected to motor driver (right)
// #define MOTOR_IN4_PIN          GPIO_NUM_5  // IN4 pin connected to motor driver (right)
// #define MOTOR_ENABLE_PIN_RIGHT GPIO_NUM_4  // Same as above, for right motor 


// #define MOTOR_PWM_FREQ          5000  // Frequency in Hz for PWM
// #define MOTOR_PWM_CHANNEL_RIGHT LEDC_CHANNEL_0
// #define MOTOR_PWM_CHANNEL_LEFT  LEDC_CHANNEL_1
// #define MOTOR_PWM_MODE          LEDC_LOW_SPEED_MODE
// #define MOTOR_PWM_TIMER         LEDC_TIMER_0
// #define MOTOR_PWM_RES           LEDC_TIMER_10_BIT  // PWM resolution (10-bit)
// #define MAX_DUTY_CYCLE          1023  // Maximum duty cycle for 10-bit resolution (1023)

// // Testing, get rid of this, not needed anymore
// // int direction_for_left = 1;  // 1 for clockwise, 0 for anticlockwise  
// //                              // 1 for forward, 0 for backwards
// // int direction_for_right = 1; // 1 for anticlockwise, 0 for clockwise 
// //                              // same as direction for left, 1 forward, zero backwards

// // Speed control
// int duty_cycle = 700;  //1023 is max it can go, 
//                        // after testing it seems that 700 is the lowest it can go while still functioning normally

// void stop() {
//     // set and apply via update. Set pwn to 0 for either stop or resetting purposes
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, 0);  
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, 0);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
// }

// void enable_motors() {
//     // set and apply via update
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, duty_cycle);
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, duty_cycle);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
// }

// void backward() {
//     // Set spin diretion to backward
//     stop();
//     gpio_set_level(MOTOR_IN3_PIN, 0);
//     gpio_set_level(MOTOR_IN4_PIN, 1);
//     gpio_set_level(MOTOR_IN1_PIN, 0);
//     gpio_set_level(MOTOR_IN2_PIN, 1);
//     enable_motors();
// } 
// void forward() {
//     // Set spin diretion to forward
//     stop();
//     gpio_set_level(MOTOR_IN3_PIN, 1);
//     gpio_set_level(MOTOR_IN4_PIN, 0);
//     gpio_set_level(MOTOR_IN1_PIN, 1);
//     gpio_set_level(MOTOR_IN2_PIN, 0);
//     enable_motors();
// }

// void left() {
//     // Move right wheel only to turn left
//     stop(); 
//     gpio_set_level(MOTOR_IN3_PIN, 1);
//     gpio_set_level(MOTOR_IN4_PIN, 0);
//     gpio_set_level(MOTOR_IN1_PIN, 0);
//     gpio_set_level(MOTOR_IN2_PIN, 0);

//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, duty_cycle);  
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, 0);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
// }

// void right() {
//     // Move left wheel only to turn right
//     stop();
//     gpio_set_level(MOTOR_IN3_PIN, 0);
//     gpio_set_level(MOTOR_IN4_PIN, 0);
//     gpio_set_level(MOTOR_IN1_PIN, 1);
//     gpio_set_level(MOTOR_IN2_PIN, 0);

//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, 0);  
//     ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, duty_cycle);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
//     ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
// }


// void app_main(void) {
//     // Configure GPIO for motor direction control
//     esp_rom_gpio_pad_select_gpio(MOTOR_IN1_PIN);
//     gpio_set_direction(MOTOR_IN1_PIN, GPIO_MODE_OUTPUT);
//     esp_rom_gpio_pad_select_gpio(MOTOR_IN2_PIN);
//     gpio_set_direction(MOTOR_IN2_PIN, GPIO_MODE_OUTPUT);

//     esp_rom_gpio_pad_select_gpio(MOTOR_IN3_PIN);
//     gpio_set_direction(MOTOR_IN3_PIN, GPIO_MODE_OUTPUT);
//     esp_rom_gpio_pad_select_gpio(MOTOR_IN4_PIN);
//     gpio_set_direction(MOTOR_IN4_PIN, GPIO_MODE_OUTPUT);


//     // Configure PWM timer
//     ledc_timer_config_t pwm_timer = {
//         .speed_mode       = MOTOR_PWM_MODE,
//         .duty_resolution  = MOTOR_PWM_RES,
//         .timer_num        = MOTOR_PWM_TIMER,
//         .freq_hz          = MOTOR_PWM_FREQ,
//         .clk_cfg          = LEDC_AUTO_CLK
//     };
//     ledc_timer_config(&pwm_timer);

//     // Configure PWM channel right wheel
//     ledc_channel_config_t pwm_channel_right = {
//         .gpio_num       = MOTOR_ENABLE_PIN_RIGHT,
//         .speed_mode     = MOTOR_PWM_MODE,
//         .channel        = LEDC_CHANNEL_0,
//         .intr_type      = LEDC_INTR_DISABLE,
//         .timer_sel      = MOTOR_PWM_TIMER,
//         .duty           = 0,
//         .hpoint         = 0
//     };
//     ledc_channel_config(&pwm_channel_right);

//     // Conifgure PWM channel left wheel
//     ledc_channel_config_t pwm_channel_left = {
//         .gpio_num       = MOTOR_ENABLE_PIN_LEFT,
//         .speed_mode     = MOTOR_PWM_MODE,
//         .channel        = LEDC_CHANNEL_1,
//         .intr_type      = LEDC_INTR_DISABLE,
//         .timer_sel      = MOTOR_PWM_TIMER,
//         .duty           = 0,
//         .hpoint         = 0
//     };
//     ledc_channel_config(&pwm_channel_left);

//     // give hardware time to stabilize/initialize everything
//     vTaskDelay(pdMS_TO_TICKS(500)); 

//     // Note: speed control added (speed control uses ledc)

//     while (true) {

//         stop();
//         char key = getchar(); 

//         // Clear buff of everything received, read most recent
//         int ch;
//         while ((ch = fgetc(stdin)) != EOF);

//         // Debug print
//         // printf("Received: %c\n", key);

//         switch (key) {
//             case 'w': // Forward
//                 forward();
//                 break; 

//             case 'a': // Left
//                 left();
//                 break; 

//             case 's': // Backward 
//                 backward();
//                 break; 

//             case 'd': // Right
//                 right();
//                 break; 

//             default: 
//                 stop();
//                 break; 
            
//         }      

//         if (key == 'a' || key == 'd') {
//             // Shorter delay for more controlled turns
//             vTaskDelay(pdMS_TO_TICKS(200)); 
//         }
//         else {
//             vTaskDelay(pdMS_TO_TICKS(500)); 
//         }
//     }
            

// }
