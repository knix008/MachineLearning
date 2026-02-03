/**
 * ZKTeco Push Protocol - Device Simulator (Client)
 *
 * This program simulates a ZKTeco biometric device that pushes
 * attendance records to a server using the ZKTeco Push Protocol.
 *
 * Compile (Windows): gcc -o zkteco_sim protocol.c -lws2_32
 * Compile (Linux):   gcc -o zkteco_sim protocol.c
 *
 * Usage: zkteco_sim <server_ip> <server_port>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define closesocket close
#endif

#define BUFFER_SIZE 4096
#define MAX_RECORDS 100

/* Device configuration */
typedef struct {
    char serial_number[32];
    char device_name[64];
    char firmware_version[16];
    char mac_address[18];
    int user_count;
    int fp_count;          /* fingerprint count */
    int face_count;
    int transaction_count;
} DeviceConfig;

/* Attendance record */
typedef struct {
    char user_id[24];
    char timestamp[20];    /* YYYY-MM-DD HH:MM:SS */
    int punch_type;        /* 0=Check-In, 1=Check-Out, 2=Break-Out, 3=Break-In, 4=OT-In, 5=OT-Out */
    int verify_type;       /* 0=Password, 1=Fingerprint, 2=Card, 9=Face */
    int work_code;
} AttendanceRecord;

/* Global variables */
static DeviceConfig g_device;
static char g_server_ip[64] = "127.0.0.1";
static int g_server_port = 8080;

/* Function prototypes */
int init_network(void);
void cleanup_network(void);
SOCKET create_connection(const char *host, int port);
int send_http_request(SOCKET sock, const char *method, const char *path,
                      const char *content_type, const char *body);
int receive_http_response(SOCKET sock, char *buffer, int buffer_size);
void init_device_config(DeviceConfig *config);
int device_register(void);
int device_heartbeat(void);
int send_attendance_records(AttendanceRecord *records, int count);
void generate_sample_records(AttendanceRecord *records, int count);
void print_usage(const char *program);
char *url_encode(const char *str, char *encoded, int max_len);
void get_current_timestamp(char *buffer, int size);

/**
 * Initialize network (Windows-specific WSA startup)
 */
int init_network(void) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return -1;
    }
#endif
    return 0;
}

/**
 * Cleanup network resources
 */
void cleanup_network(void) {
#ifdef _WIN32
    WSACleanup();
#endif
}

/**
 * Create TCP connection to server
 */
SOCKET create_connection(const char *host, int port) {
    SOCKET sock;
    struct sockaddr_in server_addr;
    struct hostent *server;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        fprintf(stderr, "Error: Cannot create socket\n");
        return INVALID_SOCKET;
    }

    server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Error: Cannot resolve host %s\n", host);
        closesocket(sock);
        return INVALID_SOCKET;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    server_addr.sin_port = htons(port);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        fprintf(stderr, "Error: Cannot connect to %s:%d\n", host, port);
        closesocket(sock);
        return INVALID_SOCKET;
    }

    return sock;
}

/**
 * Send HTTP request
 */
int send_http_request(SOCKET sock, const char *method, const char *path,
                      const char *content_type, const char *body) {
    char request[BUFFER_SIZE];
    int body_len = body ? (int)strlen(body) : 0;
    int request_len;

    if (body && body_len > 0) {
        request_len = snprintf(request, sizeof(request),
            "%s %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n"
            "\r\n"
            "%s",
            method, path, g_server_ip, g_server_port,
            content_type ? content_type : "application/x-www-form-urlencoded",
            body_len, body);
    } else {
        request_len = snprintf(request, sizeof(request),
            "%s %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Connection: close\r\n"
            "\r\n",
            method, path, g_server_ip, g_server_port);
    }

    if (send(sock, request, request_len, 0) < 0) {
        fprintf(stderr, "Error: Failed to send request\n");
        return -1;
    }

    printf(">>> Sent %s request to %s\n", method, path);
    return 0;
}

/**
 * Receive HTTP response
 */
int receive_http_response(SOCKET sock, char *buffer, int buffer_size) {
    int total_received = 0;
    int received;

    memset(buffer, 0, buffer_size);

    while ((received = recv(sock, buffer + total_received,
                           buffer_size - total_received - 1, 0)) > 0) {
        total_received += received;
        if (total_received >= buffer_size - 1) break;
    }

    buffer[total_received] = '\0';
    return total_received;
}

/**
 * URL encode a string
 */
char *url_encode(const char *str, char *encoded, int max_len) {
    static const char *hex = "0123456789ABCDEF";
    int pos = 0;

    while (*str && pos < max_len - 4) {
        if ((*str >= 'A' && *str <= 'Z') ||
            (*str >= 'a' && *str <= 'z') ||
            (*str >= '0' && *str <= '9') ||
            *str == '-' || *str == '_' || *str == '.' || *str == '~') {
            encoded[pos++] = *str;
        } else if (*str == ' ') {
            encoded[pos++] = '+';
        } else {
            encoded[pos++] = '%';
            encoded[pos++] = hex[(*str >> 4) & 0x0F];
            encoded[pos++] = hex[*str & 0x0F];
        }
        str++;
    }
    encoded[pos] = '\0';
    return encoded;
}

/**
 * Get current timestamp in ZKTeco format
 */
void get_current_timestamp(char *buffer, int size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

/**
 * Initialize device configuration with default values
 */
void init_device_config(DeviceConfig *config) {
    strcpy(config->serial_number, "ZKTEST123456");
    strcpy(config->device_name, "ZKTeco_Simulator");
    strcpy(config->firmware_version, "Ver 6.60");
    strcpy(config->mac_address, "00:17:61:AA:BB:CC");
    config->user_count = 100;
    config->fp_count = 150;
    config->face_count = 50;
    config->transaction_count = 0;
}

/**
 * Register device with server (sendinfo command)
 * This is called when device first connects to server
 */
int device_register(void) {
    SOCKET sock;
    char path[512];
    char body[BUFFER_SIZE];
    char response[BUFFER_SIZE];
    int result = -1;

    printf("\n=== Device Registration ===\n");

    /* Build registration path - ZKTeco push protocol format */
    snprintf(path, sizeof(path),
        "/iclock/cdata?SN=%s&options=all&pushver=2.4.1&language=en",
        g_device.serial_number);

    /* Build device info body */
    snprintf(body, sizeof(body),
        "~DeviceName=%s\r\n"
        "MAC=%s\r\n"
        "TransactionCount=%d\r\n"
        "UserCount=%d\r\n"
        "FPCount=%d\r\n"
        "FaceCount=%d\r\n"
        "FWVersion=%s\r\n"
        "PushVersion=2.4.1\r\n",
        g_device.device_name,
        g_device.mac_address,
        g_device.transaction_count,
        g_device.user_count,
        g_device.fp_count,
        g_device.face_count,
        g_device.firmware_version);

    sock = create_connection(g_server_ip, g_server_port);
    if (sock == INVALID_SOCKET) {
        return -1;
    }

    if (send_http_request(sock, "POST", path, "text/plain", body) == 0) {
        if (receive_http_response(sock, response, sizeof(response)) > 0) {
            printf("<<< Server Response:\n%s\n", response);
            result = 0;
        }
    }

    closesocket(sock);
    return result;
}

/**
 * Send heartbeat to server (getrequest command)
 * Device periodically calls this to check for pending commands
 */
int device_heartbeat(void) {
    SOCKET sock;
    char path[512];
    char response[BUFFER_SIZE];
    int result = -1;

    printf("\n=== Device Heartbeat ===\n");

    /* Build heartbeat path */
    snprintf(path, sizeof(path),
        "/iclock/getrequest?SN=%s",
        g_device.serial_number);

    sock = create_connection(g_server_ip, g_server_port);
    if (sock == INVALID_SOCKET) {
        return -1;
    }

    if (send_http_request(sock, "GET", path, NULL, NULL) == 0) {
        if (receive_http_response(sock, response, sizeof(response)) > 0) {
            printf("<<< Server Response:\n%s\n", response);
            /* Parse and handle any commands from server here */
            result = 0;
        }
    }

    closesocket(sock);
    return result;
}

/**
 * Send attendance records to server (cdata with ATTLOG table)
 */
int send_attendance_records(AttendanceRecord *records, int count) {
    SOCKET sock;
    char path[512];
    char body[BUFFER_SIZE];
    char response[BUFFER_SIZE];
    char record_line[256];
    int i;
    int result = -1;

    if (count <= 0 || records == NULL) {
        return -1;
    }

    printf("\n=== Sending %d Attendance Record(s) ===\n", count);

    /* Build path for attendance log upload */
    snprintf(path, sizeof(path),
        "/iclock/cdata?SN=%s&table=ATTLOG&Stamp=0",
        g_device.serial_number);

    /* Build attendance records body
     * Format: PIN\tTIME\tSTATUS\tVERIFY\tWORKCODE\tReserved1\tReserved2 */
    body[0] = '\0';
    for (i = 0; i < count; i++) {
        snprintf(record_line, sizeof(record_line),
            "%s\t%s\t%d\t%d\t%d\t\t\r\n",
            records[i].user_id,
            records[i].timestamp,
            records[i].punch_type,
            records[i].verify_type,
            records[i].work_code);

        if (strlen(body) + strlen(record_line) < sizeof(body) - 1) {
            strcat(body, record_line);
        } else {
            fprintf(stderr, "Warning: Record buffer full, some records not sent\n");
            break;
        }

        printf("  Record %d: User=%s, Time=%s, Type=%d, Verify=%d\n",
               i + 1, records[i].user_id, records[i].timestamp,
               records[i].punch_type, records[i].verify_type);
    }

    sock = create_connection(g_server_ip, g_server_port);
    if (sock == INVALID_SOCKET) {
        return -1;
    }

    if (send_http_request(sock, "POST", path, "text/plain", body) == 0) {
        if (receive_http_response(sock, response, sizeof(response)) > 0) {
            printf("<<< Server Response:\n%s\n", response);
            g_device.transaction_count += count;
            result = 0;
        }
    }

    closesocket(sock);
    return result;
}

/**
 * Generate sample attendance records for testing
 */
void generate_sample_records(AttendanceRecord *records, int count) {
    int i;
    char timestamp[20];
    time_t now = time(NULL);
    struct tm *tm_info;

    for (i = 0; i < count; i++) {
        /* Generate user ID (1-100) */
        snprintf(records[i].user_id, sizeof(records[i].user_id), "%d", (i % 100) + 1);

        /* Generate timestamp - offset by minutes for variety */
        tm_info = localtime(&now);
        tm_info->tm_min -= (count - i);
        mktime(tm_info);  /* normalize the time */
        strftime(records[i].timestamp, sizeof(records[i].timestamp),
                 "%Y-%m-%d %H:%M:%S", tm_info);

        /* Alternate between check-in (0) and check-out (1) */
        records[i].punch_type = i % 2;

        /* Use fingerprint (1) as default verify type */
        records[i].verify_type = 1;

        /* No work code */
        records[i].work_code = 0;
    }
}

/**
 * Print usage instructions
 */
void print_usage(const char *program) {
    printf("ZKTeco Push Protocol - Device Simulator\n");
    printf("========================================\n\n");
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  -h, --host <ip>      Server IP address (default: 127.0.0.1)\n");
    printf("  -p, --port <port>    Server port (default: 8080)\n");
    printf("  -s, --serial <sn>    Device serial number (default: ZKTEST123456)\n");
    printf("  -r, --records <n>    Number of sample records to send (default: 5)\n");
    printf("  -i, --interactive    Run in interactive mode\n");
    printf("  --help               Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -h 192.168.1.100 -p 8080\n", program);
    printf("  %s -s DEVICE001 -r 10\n", program);
}

/**
 * Interactive menu
 */
void interactive_mode(void) {
    int choice;
    int record_count;
    AttendanceRecord records[MAX_RECORDS];
    char user_id[24];
    char input[64];

    printf("\n");
    printf("==============================================\n");
    printf("  ZKTeco Device Simulator - Interactive Mode  \n");
    printf("==============================================\n");
    printf("Server: %s:%d\n", g_server_ip, g_server_port);
    printf("Device SN: %s\n", g_device.serial_number);
    printf("==============================================\n");

    while (1) {
        printf("\n--- Menu ---\n");
        printf("1. Register device with server\n");
        printf("2. Send heartbeat (check for commands)\n");
        printf("3. Send sample attendance records\n");
        printf("4. Send custom attendance record\n");
        printf("5. Change server settings\n");
        printf("6. Change device serial number\n");
        printf("0. Exit\n");
        printf("Choice: ");

        if (fgets(input, sizeof(input), stdin) == NULL) break;
        choice = atoi(input);

        switch (choice) {
            case 1:
                device_register();
                break;

            case 2:
                device_heartbeat();
                break;

            case 3:
                printf("Number of records to send (1-%d): ", MAX_RECORDS);
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                record_count = atoi(input);
                if (record_count < 1) record_count = 1;
                if (record_count > MAX_RECORDS) record_count = MAX_RECORDS;
                generate_sample_records(records, record_count);
                send_attendance_records(records, record_count);
                break;

            case 4:
                printf("Enter user ID: ");
                if (fgets(user_id, sizeof(user_id), stdin) == NULL) break;
                user_id[strcspn(user_id, "\r\n")] = '\0';

                memset(&records[0], 0, sizeof(AttendanceRecord));
                strncpy(records[0].user_id, user_id, sizeof(records[0].user_id) - 1);
                get_current_timestamp(records[0].timestamp, sizeof(records[0].timestamp));

                printf("Punch type (0=In, 1=Out, 2=Break-Out, 3=Break-In): ");
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                records[0].punch_type = atoi(input);

                printf("Verify type (0=Password, 1=Fingerprint, 2=Card, 9=Face): ");
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                records[0].verify_type = atoi(input);

                records[0].work_code = 0;
                send_attendance_records(records, 1);
                break;

            case 5:
                printf("Enter server IP: ");
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                input[strcspn(input, "\r\n")] = '\0';
                if (strlen(input) > 0) {
                    strncpy(g_server_ip, input, sizeof(g_server_ip) - 1);
                }
                printf("Enter server port: ");
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                if (atoi(input) > 0) {
                    g_server_port = atoi(input);
                }
                printf("Updated: %s:%d\n", g_server_ip, g_server_port);
                break;

            case 6:
                printf("Enter device serial number: ");
                if (fgets(input, sizeof(input), stdin) == NULL) break;
                input[strcspn(input, "\r\n")] = '\0';
                if (strlen(input) > 0) {
                    strncpy(g_device.serial_number, input, sizeof(g_device.serial_number) - 1);
                }
                printf("Device SN updated: %s\n", g_device.serial_number);
                break;

            case 0:
                printf("Exiting...\n");
                return;

            default:
                printf("Invalid choice\n");
        }
    }
}

/**
 * Main entry point
 */
int main(int argc, char *argv[]) {
    int i;
    int record_count = 5;
    int interactive = 0;
    AttendanceRecord records[MAX_RECORDS];

    /* Initialize device with defaults */
    init_device_config(&g_device);

    /* Parse command line arguments */
    for (i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--host") == 0) && i + 1 < argc) {
            strncpy(g_server_ip, argv[++i], sizeof(g_server_ip) - 1);
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc) {
            g_server_port = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--serial") == 0) && i + 1 < argc) {
            strncpy(g_device.serial_number, argv[++i], sizeof(g_device.serial_number) - 1);
        }
        else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--records") == 0) && i + 1 < argc) {
            record_count = atoi(argv[++i]);
            if (record_count > MAX_RECORDS) record_count = MAX_RECORDS;
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactive = 1;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Initialize networking */
    if (init_network() != 0) {
        return 1;
    }

    if (interactive) {
        interactive_mode();
    } else {
        /* Default flow: Register -> Send records -> Heartbeat */
        printf("ZKTeco Push Protocol - Device Simulator\n");
        printf("Server: %s:%d\n", g_server_ip, g_server_port);
        printf("Device SN: %s\n\n", g_device.serial_number);

        /* Step 1: Register device */
        if (device_register() != 0) {
            fprintf(stderr, "Warning: Device registration failed\n");
        }

        /* Step 2: Send sample attendance records */
        generate_sample_records(records, record_count);
        if (send_attendance_records(records, record_count) != 0) {
            fprintf(stderr, "Warning: Failed to send attendance records\n");
        }

        /* Step 3: Send heartbeat */
        if (device_heartbeat() != 0) {
            fprintf(stderr, "Warning: Heartbeat failed\n");
        }
    }

    cleanup_network();
    return 0;
}
