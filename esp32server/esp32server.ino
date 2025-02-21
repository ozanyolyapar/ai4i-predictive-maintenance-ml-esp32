#include <WiFi.h>
#include <WebServer.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const char* ssid = "WIN-1VOA252UCBG8420";
const char* password = "14a1}O99";

// StandardScaler parameters, exported in jupyter notebook
float means[] = {310.01306712466004, 1538.68770573923, 40.0256476313153, 107.80964648633176, 9.9970087304995, 6285.8163749150835, 4316.1199370259055};
float std_devs[] = {1.4828805521220576, 179.488094417115, 9.96590627149833, 63.576480855468624, 1.0009189625018806, 1069.5033667515468, 2838.9191488600954};

const char* index_html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 MLP Inference</title>
  <script>
    let timeout;
    function updateIframe() {
      clearTimeout(timeout)
      let qs = '?pt=' + document.getElementById('pt').value +
               '&rs=' + document.getElementById('rs').value +
               '&tq=' + document.getElementById('tq').value +
               '&tw=' + document.getElementById('tw').value +
               '&td=' + document.getElementById('td').value +
               '&mp=' + document.getElementById('mp').value +
               '&tl=' + document.getElementById('tl').value +
               '&type=' + document.querySelector('input[name=type]:checked').value;
      timeout = setTimeout(function() {document.getElementById('resultFrame').src = '/infer' + qs;}, 3000)
    }
  </script>
</head>
<body>
  <h2>MLP Inference</h2>
  <form oninput="updateIframe()">
    Process Temp: <input type="range" id="pt" min="290" max="330" value="310"><br>
    Rotational Speed: <input type="range" id="rs" min="1000" max="3000" value="2000"><br>
    Torque: <input type="range" id="tq" min="0" max="90" value="45"><br>
    Tool Wear: <input type="range" id="tw" min="0" max="250" value="125"><br>
    Temp Diff: <input type="range" id="td" min="0" max="20" value="10"><br>
    Mech Power: <input type="range" id="mp" min="800" max="11000" value="5900"><br>
    Tool Load: <input type="range" id="tl" min="8000" max="15000" value="11500"><br>
    Type: <input type="radio" name="type" value="L" checked> L 
          <input type="radio" name="type" value="M"> M 
          <input type="radio" name="type" value="H"> H <br>
  </form>
  <iframe id="resultFrame" width="100%" height="50"></iframe>
</body>
</html>
)rawliteral";

WebServer server(80);

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
const tflite::Model* model = nullptr;

// tensor arena allocation
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

#include "mlp_tf.h"  // Ensure mlp_tf.h is in your project folder

void loadModel() {
  model = tflite::GetModel(mlp_tf_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }
  
  static tflite::AllOpsResolver resolver;
  
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  Serial.println("Model loaded and interpreter initialized.");
}

void handleRoot() {
  server.send(200, "text/html", index_html);
}

void handleInfer() {
  if (!server.hasArg("pt") || !server.hasArg("rs") || !server.hasArg("tq") ||
      !server.hasArg("tw") || !server.hasArg("td") || !server.hasArg("mp") ||
      !server.hasArg("tl") || !server.hasArg("type")) {
    server.send(400, "text/plain", "Missing parameters");
    return;
  }
  
  float input_vals[9];
  const char* params[7] = {"pt", "rs", "tq", "tw", "td", "mp", "tl"};
  for (int i = 0; i < 7; i++) {
    float val = server.arg(params[i]).toFloat();
    input_vals[i] = (val - means[i]) / std_devs[i];
  }
  
  String typeStr = server.arg("type");
  input_vals[7] = (typeStr == "M") ? 1.0f : 0.0f;
  input_vals[8] = (typeStr == "H") ? 1.0f : 0.0f;
  
  // Copy the input vector into the model's input tensor
  for (int i = 0; i < 9; i++) {
    input_tensor->data.f[i] = input_vals[i];
  }
  
  uint32_t start_time = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    server.send(500, "text/plain", "Model inference failed");
    return;
  }
  uint32_t inference_time = micros() - start_time;
  
  float result = output_tensor->data.f[0];
  
  String response = "Inference Time (us): " + String(inference_time) + "\nResult: " + String(result);
  server.send(200, "text/plain", response);
}

void setup() {
  Serial.begin(115200);
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
  
  loadModel();
  
  server.on("/", handleRoot);
  server.on("/infer", handleInfer);
  server.begin();
  Serial.println("Web server started.");
}

void loop() {
  server.handleClient();
}
