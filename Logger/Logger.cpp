#include <stdio.h>
#include <time.h>

#include "Logger.h"
#include "util.h"

bool LogLevel::isValid(int level)
{
	switch(level) {
		case Info:
		case Error:
		case Debug:
			return true;
		default:
			return false;
	}
}

std::string LogLevel::toString(int level)
{
	switch(level) {
		case Info:
			return "Info";
		case Error:
			return "Error";
		case Debug:
			return "Debug";
        case Warning:
            return "Warning";
	}

	return "";
}

// getDateTimeString and formatLog are no longer needed as spdlog handles formatting.
/*
std::string Logger::getDateTimeString() { ... }
std::string Logger::formatLog(int logLevel, std::string msg) { ... }
*/

void Logger::log(int logLevel, std::string msg)
{
    // spdlog's default logger is already set up in main.cpp
    // It includes timestamp, level, thread ID.
    // We just need to map our LogLevel to spdlog's levels.
    switch (logLevel) {
        case LogLevel::Info:
            spdlog::info(msg);
            break;
        case LogLevel::Error:
            spdlog::error(msg);
            break;
        case LogLevel::Debug:
            spdlog::debug(msg);
            break;
        case LogLevel::Warning: // Assuming spdlog::warn exists
            spdlog::warn(msg);
            break;
        default:
            // Handle unknown log level, perhaps log as info or warning
            spdlog::warn("Unknown log level ({}): {}", logLevel, msg);
            break;
    }
}

void Logger::setLogFile(std::string path)
{
    // This function is currently a no-op.
    // If file logging is desired with spdlog, it should be configured
    // in SetupGlobalLogging() in main.cpp by adding a file sink.
    // For example:
    // auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
    // spdlog::default_logger()->sinks().push_back(file_sink);
    // Or create a new logger for the file:
    // auto file_logger = spdlog::basic_logger_mt("file_logger", path);
    // file_logger->info("This message goes to file");
    //
    // For now, this C++ Logger class won't manage spdlog sinks directly.
    // That's centralized in main.cpp's SetupGlobalLogging.
    spdlog::info("Logger::setLogFile called with path '{}', but spdlog file sink setup is centralized.", path);
}
