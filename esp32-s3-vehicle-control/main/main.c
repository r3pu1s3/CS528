#include <stdio.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include <driver/ledc.h>
#include "driver/gpio.h"  
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

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

// Testing, get rid of this, not needed anymore
// int direction_for_left = 1;  // 1 for clockwise, 0 for anticlockwise  
//                              // 1 for forward, 0 for backwards
// int direction_for_right = 1; // 1 for anticlockwise, 0 for clockwise 
//                              // same as direction for left, 1 forward, zero backwards

// Speed control
int duty_cycle = 700;  //1023 is max it can go, 
                       // after testing it seems that 700 is the lowest it can go while still functioning normally

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
    gpio_set_level(MOTOR_IN3_PIN, 0);
    gpio_set_level(MOTOR_IN4_PIN, 1);
    gpio_set_level(MOTOR_IN1_PIN, 0);
    gpio_set_level(MOTOR_IN2_PIN, 1);
    enable_motors();
} 
void forward() {
    // Set spin diretion to forward
    stop();
    gpio_set_level(MOTOR_IN3_PIN, 1);
    gpio_set_level(MOTOR_IN4_PIN, 0);
    gpio_set_level(MOTOR_IN1_PIN, 1);
    gpio_set_level(MOTOR_IN2_PIN, 0);
    enable_motors();
}

void left() {
    // Move right wheel only to turn left
    stop(); 
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
    gpio_set_level(MOTOR_IN3_PIN, 0);
    gpio_set_level(MOTOR_IN4_PIN, 0);
    gpio_set_level(MOTOR_IN1_PIN, 1);
    gpio_set_level(MOTOR_IN2_PIN, 0);

    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT, 0);  
    ledc_set_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT, duty_cycle);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_RIGHT);
    ledc_update_duty(MOTOR_PWM_MODE, MOTOR_PWM_CHANNEL_LEFT);
}


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

    // Note: speed control added (speed control uses ledc)

    while (true) {

        stop();
        char key = getchar(); 

        // Clear buff of everything received, read most recent
        int ch;
        while ((ch = fgetc(stdin)) != EOF);

        // Debug print
        // printf("Received: %c\n", key);

        switch (key) {
            case 'w': // Forward
                forward();
                break; 

            case 'a': // Left
                left();
                break; 

            case 's': // Backward 
                backward();
                break; 

            case 'd': // Right
                right();
                break; 

            default: 
                stop();
                break; 
            
        }      

        if (key == 'a' || key == 'd') {
            // Shorter delay for more controlled turns
            vTaskDelay(pdMS_TO_TICKS(200)); 
        }
        else {
            vTaskDelay(pdMS_TO_TICKS(500)); 
        }
    }
            

}
