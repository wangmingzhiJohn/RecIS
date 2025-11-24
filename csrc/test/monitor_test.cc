/* Test Command:
#!/bin/bash
cd ..; rm -rf ./log/ ./logs/
g++ -g -Wall -Werror -o test/monitor_test.bin -lpthread -I./ \
    test/monitor_test.cc monitor/log_writer.c monitor/metric_client.cc
./test/monitor_test.bin
 */

// Include standard headers
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>

// Include project headers
#include "monitor/metric_client.h"

using namespace recis::monitor;
using namespace std;

// Helper function to compare floating point numbers
bool approx_equal(double a, double b, double epsilon = 1e-6) {
  return std::abs(a - b) < epsilon;
}

void test_point_tag_creation() {
  cout << "-- Test PointTag creation..." << endl;

  // Test basic creation
  map<string, string> tags_map = {{"service", "test"}, {"env", "dev"}};
  shared_ptr<PointTag> tag = make_shared<PointTag>(tags_map);

  // Test cloning
  map<string, string> extra_tags = {{"version", "1.0"}};
  shared_ptr<PointTag> cloned_tag1 = PointTag::Clone(tag);
  shared_ptr<PointTag> cloned_tag2 = PointTag::Clone(nullptr, extra_tags);

  assert(cloned_tag1 != nullptr);
  assert(cloned_tag2 != nullptr);
  assert(*cloned_tag1 == *tag);
  assert(*cloned_tag2 != *tag);
  cout << "-- Pass PointTag creation and cloning ✅" << endl;
}

void test_factory_creation() {
  cout << "-- Test Factory creation..." << endl;

  // Create factory with 1 second interval
  auto factory = Factory::MakeInstance(chrono::seconds(1));
  assert(factory != nullptr);

  // Get a client
  auto client = factory->get_client("test_client");
  assert(client != nullptr);

  auto& factory_static = Factory::StaticInstance();
  assert(factory_static != nullptr);
  auto client2 = factory_static->get_client("test_client_static");

  factory.reset();
  factory_static.reset();  // 测试静态成员退出析构是否正常
  cout << "-- Pass Factory creation ✅" << endl;
}

void test_factory_del_immediatly() {
  cout << "-- Test Factory delete immediately..." << endl;
  system("rm -f ./log/monitor/metric_immediate.log*");
  auto factory = Factory::MakeInstance("./log/monitor/metric_immediate.log",
                                       chrono::seconds(30));

  // Get a client
  auto client = factory->get_client("test_client_creat_immediate");
  assert(client != nullptr);
  map<string, string> tag_map = {{"endpoint", "test_immediate"}};
  auto tag = make_shared<PointTag>(tag_map);
  bool result = client->report("response_time", 100.01, tag, PointType::kGauge);
  assert(result);

  auto start = chrono::system_clock::now();
  factory.reset();
  auto end = chrono::system_clock::now();
  assert(end - start < chrono::seconds(1));

  // grep test_immediate from log/monitor/metric_immediate.log, and assert(grep
  // wc -l not 0)
  int count = 0;
  string cmd =
      "grep test_client_creat_immediate ./log/monitor/metric_immediate.log | "
      "wc -l";
  FILE* fp = popen(cmd.c_str(), "r");
  assert(fp != nullptr);
  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), fp)) {
    count += atoi(buffer);
  }
  pclose(fp);
  fp = nullptr;

  cout << "-- Pass Factory delete immediately ✅ write_count: " << count
       << endl;
}

void test_client_report_gauge() {
  cout << "-- Test Client Gauge reporting..." << endl;

  auto factory = Factory::MakeInstance(chrono::seconds(1));
  auto client = factory->get_client("test_gauge_client");

  // Create tags
  map<string, string> tag_map = {{"endpoint", "api/v1/test"}};
  auto tag = make_shared<PointTag>(tag_map);

  // Report gauge values
  bool result = client->report("response_time", 100.5, tag, PointType::kGauge);
  assert(result);

  result = client->report("response_time", 200.7, tag, PointType::kGauge);
  assert(result);

  factory.reset();
  cout << "-- Pass Gauge reporting ✅ result: " << result << endl;
}

void test_client_report_counter() {
  cout << "-- Test Client Counter reporting..." << endl;

  auto factory = Factory::MakeInstance(chrono::seconds(1));
  auto client = factory->get_client("test_counter_client");

  // Create tags
  map<string, string> tag_map = {{"operation", "increment"}};
  auto tag = make_shared<PointTag>(tag_map);

  // Report counter values
  bool result =
      client->report("requests_total", int64_t(1), tag, PointType::kCounter);
  assert(result);

  result =
      client->report("requests_total", int64_t(1), tag, PointType::kCounter);
  assert(result);

  result =
      client->report("requests_total", int64_t(1), tag, PointType::kCounter);
  assert(result);

  factory.reset();
  cout << "-- Pass Counter reporting ✅ result: " << result << endl;
}

void test_client_report_summary() {
  cout << "-- Test Client Summary reporting..." << endl;

  auto factory = Factory::MakeInstance(chrono::seconds(1));
  auto client = factory->get_client("test_summary_client");

  // Create tags
  map<string, string> tag_map = {{"method", "GET"}};
  auto tag = make_shared<PointTag>(tag_map);

  // Report summary values
  bool result =
      client->report("request_duration", 50.0, tag, PointType::kSummary);
  assert(result);

  result = client->report("request_duration", 100.0, tag, PointType::kSummary);
  assert(result);

  result = client->report("request_duration", 150.0, tag, PointType::kSummary);
  assert(result);

  factory.reset();
  cout << "-- Pass Summary reporting ✅ result: " << result << endl;
}

void test_multithreaded_reporting() {
  cout << "-- Test multithreaded reporting..." << endl;

  auto factory = Factory::MakeInstance(chrono::seconds(1));
  auto client = factory->get_client("test_mt_client");

  // Create tags
  map<string, string> tag_map = {{"component", "multithread_test"}};
  auto tag = make_shared<PointTag>(tag_map);

  // Launch multiple threads to report metrics
  const int num_threads = 10;
  const int reports_per_thread = 12 * 1000;
  vector<thread> threads;

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&client, &tag, t]() {
      for (int i = 0; i < reports_per_thread; i++) {
        client->report("mt_counter", int64_t(t * reports_per_thread + i), tag,
                       PointType::kCounter);
        this_thread::sleep_for(chrono::milliseconds(1));  // Small delay
      }
    });
  }

  // Wait for all threads to complete
  for (auto& th : threads) {
    th.join();
  }

  factory.reset();
  cout << "-- Pass Multithreaded reporting ✅" << endl;
}

void test_multifactory_multiclient_reporting() {
  cout << "-- Test multifactory reporting..." << endl;
  vector<unique_ptr<Factory>> factories;
  bool result;
  for (int i = 0; i < 5; i++) {  // 5 factory
    factories.emplace_back(nullptr);
    factories[i] = Factory::MakeInstance(chrono::seconds(1));
    auto& factory = factories[i];
    auto client = factory->get_client("test_mf_client_" + to_string(i));

    // Create tags
    map<string, string> tag_map = {{"component", "multifactory_test"}};
    auto tag = make_shared<PointTag>(tag_map);

    // Report metrics
    result = client->report("mf_counter", int64_t(i), tag, PointType::kCounter);
    assert(result);
  }
  for (unique_ptr<Factory>& factory : factories) {
    factory.reset();
  }
  cout << "-- Pass Multifactory reporting ✅ result: " << result << endl;
}

int main() {
  cout << "=== ⏩: Starting Metrics Client Unit Tests 测试指标客户端单元 ..."
       << endl;

  try {
    test_point_tag_creation();
    test_factory_creation();
    test_factory_del_immediatly();
    test_client_report_gauge();
    test_client_report_counter();
    test_client_report_summary();
    test_multithreaded_reporting();
    test_multifactory_multiclient_reporting();

    cout << "=== ✅: Metrics Client Unit Tests All Passed 测试客户端单元完毕"
         << endl;
    return 0;
  } catch (const exception& e) {
    cerr << "=== ❌: Metrics Client Unit Tests failed with exception: "
         << e.what() << endl;
    return 1;
  }
}