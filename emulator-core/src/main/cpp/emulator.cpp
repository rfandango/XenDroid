// SPDX-License-Identifier: WTFPL

#include "emulator.h"
#include <android/log.h>
#include <fstream>
#include <jni.h>
#include <thread>

#define LOG_TAG "Emulator_Config"
#define LOGE(...) {      \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,"%s : %d",__FILE__,__LINE__);\
	__android_log_print(ANDROID_LOG_ERROR, LOG_TAG,__VA_ARGS__);\
}

#define CFG_TYPE_TOML 1
#define CFG_TYPE_YAML 0

#if CFG_TYPE_YAML
#include "yaml-cpp/yaml.h"

static_assert(sizeof(YAML::Node*)==8,"");

static YAML::Node* j_open_config_str(JNIEnv* env,jobject self,jstring jconfig_str){
    jboolean is_copy=false;
    const char* config_str=env->GetStringUTFChars(jconfig_str,&is_copy);
    std::istringstream config_stream(config_str);
    YAML::Node* config_node = new YAML::Node(YAML::Load(config_stream));
    env->ReleaseStringUTFChars(jconfig_str,config_str);
    return config_node;
}

static jstring j_close_config_str(JNIEnv* env,jobject self,YAML::Node* config_node){
    YAML::Emitter out;
    out << *config_node;
    jboolean is_copy=false;

    jstring result=env->NewStringUTF(out.c_str());
    delete config_node;
    return result;
}

static YAML::Node* j_open_config_file(JNIEnv* env,jobject self,jstring config_path){
    jboolean is_copy=false;
    const char* path=env->GetStringUTFChars(config_path,&is_copy);

    YAML::Node* config_node = new YAML::Node(YAML::LoadFile(path));
    env->ReleaseStringUTFChars(config_path,path);
    return config_node;
}

static jstring j_load_config_entry(JNIEnv* env,jobject self,YAML::Node* config_node,jstring tag){
    jboolean is_copy=false;
    const char* tag_cstr=env->GetStringUTFChars(tag,&is_copy);
    std::string  tag_str(tag_cstr);
    env->ReleaseStringUTFChars(tag,tag_cstr);
    size_t pos=tag_str.find('|');
    std::string  parent=tag_str.substr(0,pos);
    std::string  child=tag_str.substr(pos+1);

    switch(std::count(child.begin(),child.end(),'|')){
        case 0: {
            YAML::Node node=(*config_node)[parent][child];
            if(node.IsDefined())
                return env->NewStringUTF(node.as<std::string>().c_str());
        }
            break;
        case 1:{
            pos=child.find('|');
            std::string child2=child.substr(pos+1);
            child=child.substr(0,pos);

            YAML::Node node=(*config_node)[parent][child][child2];
            if(node.IsDefined())
                return env->NewStringUTF(node.as<std::string>().c_str());
        }
            break;

        case 2:{

            pos=child.find('|');
            std::string child2=child.substr(pos+1);
            child=child.substr(0,pos);
            pos=child2.find('|');
            std::string child3=child2.substr(pos+1);
            child2=child2.substr(0,pos);

            YAML::Node node=(*config_node)[parent][child][child2][child3];
            if(node.IsDefined())
                return env->NewStringUTF(node.as<std::string>().c_str());
        }
            break;

        default:
            LOGE("load_config_entry fail %s,level too deep",tag_str.c_str());
            break;
    }
    return NULL;
}


static jobjectArray j_load_config_entry_ty_arr(JNIEnv* env,jobject self,YAML::Node* config_node,jstring tag){
    jboolean is_copy=false;
    const char* tag_cstr=env->GetStringUTFChars(tag,&is_copy);
    std::string  tag_str(tag_cstr);
    env->ReleaseStringUTFChars(tag,tag_cstr);
    size_t pos=tag_str.find('|');
    std::string  parent=tag_str.substr(0,pos);
    std::string  child=tag_str.substr(pos+1);

    switch(std::count(child.begin(),child.end(),'|')){
        case 0: {
            YAML::Node node=(*config_node)[parent][child];
            if(node.IsDefined()) {
                std::vector<std::string> v=node.as<std::vector<std::string>>();
                jobjectArray arr=env->NewObjectArray(v.size(),env->FindClass("java/lang/String"),NULL);
                for(size_t i=0;i<v.size();i++){
                    env->SetObjectArrayElement(arr,i,env->NewStringUTF(v[i].c_str()));
                }
                return arr;
            }
        }

            break;

        default:
            LOGE("load_config_entry fail %s,level too deep",tag_str.c_str());
            break;
    }
    return NULL;
}

static void j_save_config_entry(JNIEnv* env,jobject self,YAML::Node* config_node,jstring tag,jstring val){

    jboolean is_copy=false;
    const char* tag_cstr=env->GetStringUTFChars(tag,&is_copy);
    std::string  tag_str(tag_cstr);
    env->ReleaseStringUTFChars(tag,tag_cstr);
    size_t pos=tag_str.find('|');
    std::string  parent=tag_str.substr(0,pos);
    std::string  child=tag_str.substr(pos+1);
    const char* val_cstr=env->GetStringUTFChars(val,&is_copy);

    switch(std::count(child.begin(),child.end(),'|')){
        case 0: {
            (*config_node)[parent][child] = std::string(val_cstr);
        }
            break;
        case 1:{
            pos=child.find('|');
            std::string child2=child.substr(pos+1);
            child=child.substr(0,pos);
            (*config_node)[parent][child][child2]=std::string(val_cstr);
        }
            break;

        case 2:{

            pos=child.find('|');
            std::string child2=child.substr(pos+1);
            child=child.substr(0,pos);
            pos=child2.find('|');
            std::string child3=child2.substr(pos+1);
            child2=child2.substr(0,pos);
            (*config_node)[parent][child][child2][child3]=std::string(val_cstr);
        }
            break;

        default:
            LOGE("save_config_entry fail %s,level too deep",tag_str.c_str());
            break;
    }

    env->ReleaseStringUTFChars(val,val_cstr);
}


static void j_save_config_entry_ty_arr(JNIEnv* env,jobject self,YAML::Node* config_node,jstring tag,jobjectArray val) {

    jboolean is_copy = false;
    const char *tag_cstr = env->GetStringUTFChars(tag, &is_copy);
    std::string tag_str(tag_cstr);
    env->ReleaseStringUTFChars(tag, tag_cstr);
    size_t pos = tag_str.find('|');
    std::string parent = tag_str.substr(0, pos);
    std::string child = tag_str.substr(pos + 1);

    switch (std::count(child.begin(), child.end(), '|')) {
        case 0: {
            std::vector<std::string> v;
            for (int i = 0; i < env->GetArrayLength(val); i++) {
                jstring jstr = (jstring) env->GetObjectArrayElement(val, i);
                const char *str = env->GetStringUTFChars(jstr, &is_copy);
                v.push_back(std::string(str));
                env->ReleaseStringUTFChars(jstr, str);
            }
            (*config_node)[parent][child] = v;
        }
            break;


        default:
        LOGE("save_config_entry fail %s,level too deep",tag_str.c_str());
            break;
    }
}
static void j_close_config_file(JNIEnv* env,jobject self,YAML::Node* config_node,jstring config_path){
    YAML::Emitter out;
    out << *config_node;
    jboolean is_copy=false;
    const char* path=env->GetStringUTFChars(config_path,&is_copy);
    std::ofstream fout(path);
    fout << out.c_str();
    fout.close();
    env->ReleaseStringUTFChars(config_path,path);
    delete config_node;
}
#elif CFG_TYPE_TOML
#include "tomlplusplus/include/toml++/toml.hpp"

static toml::table* j_open_config_str(JNIEnv* env, jobject self, jstring jconfig_str) {
    jboolean is_copy = false;
    const char* config_str = env->GetStringUTFChars(jconfig_str, &is_copy);

    try {
        toml::table* config_table = new toml::table(toml::parse(config_str));
        env->ReleaseStringUTFChars(jconfig_str, config_str);
        return config_table;
    } catch (const toml::parse_error& e) {
        LOGE("open_config_str parse error: %s", e.what());
        env->ReleaseStringUTFChars(jconfig_str, config_str);
        return nullptr;
    }
}

static jstring j_close_config_str(JNIEnv* env, jobject self, toml::table* config_table) {
    if (!config_table) {
        return env->NewStringUTF("{}");
    }

    std::ostringstream out;
    out << *config_table;
    delete config_table;

    return env->NewStringUTF(out.str().c_str());
}

static toml::table* j_open_config_file(JNIEnv* env, jobject self, jstring config_path) {
    jboolean is_copy = false;
    const char* path = env->GetStringUTFChars(config_path, &is_copy);

    try {
        toml::table* config_table = new toml::table(toml::parse_file(path));
        env->ReleaseStringUTFChars(config_path, path);
        return config_table;
    } catch (const toml::parse_error& e) {
        LOGE("open_config_file parse error: %s", e.what());
        env->ReleaseStringUTFChars(config_path, path);
        return nullptr;
    }
}

static jstring j_load_config_entry(JNIEnv* env, jobject self, toml::table* config_table, jstring tag) {
    if (!config_table) {
        return NULL;
    }

    jboolean is_copy = false;
    const char* tag_cstr = env->GetStringUTFChars(tag, &is_copy);
    std::string tag_str(tag_cstr);
    env->ReleaseStringUTFChars(tag, tag_cstr);

    size_t pos = tag_str.find('|');
    std::string table_name = tag_str.substr(0, pos);
    std::string key_name = tag_str.substr(pos + 1);

    toml::node* table_node_ptr = config_table->get(table_name);
    if (!table_node_ptr || !table_node_ptr->is_table()) {
        return NULL;
    }

    toml::table* nested_table = table_node_ptr->as_table();
    toml::node* value_node = nested_table->get(key_name);
    if (!value_node) {
        return NULL;
    }

    if (value_node->is_boolean()) {
        bool val = *value_node->value<bool>();
        std::string val_str = val ? "true" : "false";
        return env->NewStringUTF(val_str.c_str());
    }
    else if (value_node->is_integer()) {
        int64_t val = *value_node->value<int64_t>();
        std::string val_str = std::to_string(val);
        return env->NewStringUTF(val_str.c_str());
    }
    else if (value_node->is_floating_point()) {
        double val = *value_node->value<double>();
        std::string val_str = std::to_string(val);
        return env->NewStringUTF(val_str.c_str());
    }
    else if (value_node->is_string()) {
        std::string val = *value_node->value<std::string>();
        return env->NewStringUTF(val.c_str());
    }

    return NULL;
}
static jobjectArray j_load_config_entry_ty_arr(JNIEnv* env, jobject self, toml::table* config_table, jstring tag) {
    return NULL;
}
static void j_save_config_entry(JNIEnv* env, jobject self, toml::table* config_table, jstring tag, jstring val) {
    if (!config_table) {
        return;
    }

    jboolean is_copy = false;
    const char* tag_cstr = env->GetStringUTFChars(tag, &is_copy);
    const char* val_cstr = env->GetStringUTFChars(val, &is_copy);
    std::string tag_str(tag_cstr);
    std::string val_str(val_cstr);
    env->ReleaseStringUTFChars(tag, tag_cstr);
    env->ReleaseStringUTFChars(val, val_cstr);

    size_t pos = tag_str.find('|');
    std::string table_name = tag_str.substr(0, pos);
    std::string key_name = tag_str.substr(pos + 1);

    toml::node* table_node_ptr = config_table->get(table_name);
    toml::table* table = nullptr;

    if (table_node_ptr && table_node_ptr->is_table()) {
        table = table_node_ptr->as_table();
    } else {
        auto result = config_table->emplace<toml::table>(table_name);
        table = result.first->second.as_table();
    }

    auto is_float_number = [](const std::string& str) {
        if(std::count(str.begin(), str.end(), '.')!=1) return false;
        try{
            std::stof(str);
            return true;
        }
        catch(...){
            return false;
        }
    };

    auto is_int_number = [](const std::string& str) {
        if(str.empty()) return false;
        if(str.length()>1&&str[0]=='-')
            return std::all_of(str.begin()+1, str.end(), [](char c) {
                return std::isdigit(c);
            });
        else
            return std::all_of(str.begin(), str.end(), [](char c) {
                return std::isdigit(c);
            });
    };

    if (val_str == "true" || val_str == "false") {
        table->insert_or_assign(key_name, val_str == "true");
    }
    else if (is_float_number(val_str)) {
        table->insert_or_assign(key_name, std::stod(val_str));
    }
    else if (is_int_number(val_str)) {
        try {
            table->insert_or_assign(key_name, std::stoi(val_str));
        } catch (...) {
            table->insert_or_assign(key_name, val_str);
        }
    }
    else {
        table->insert_or_assign(key_name, val_str);
    }
}
static void j_save_config_entry_ty_arr(JNIEnv* env, jobject self, toml::table* config_table, jstring tag, jobjectArray val) {
}

static void j_close_config_file(JNIEnv* env, jobject self, toml::table* config_table, jstring config_path) {
    if (!config_table) {
        return;
    }

    jboolean is_copy = false;
    const char* path = env->GetStringUTFChars(config_path, &is_copy);

    std::ofstream fout(path);
    if (fout.is_open()) {
        fout << *config_table;
        fout.close();
    }

    env->ReleaseStringUTFChars(config_path, path);
    delete config_table;
}

static void j_free_config(JNIEnv* env, jobject self, toml::table* config_table) {
    delete config_table;
}

#else
#error "CFG_TYPE_XXX not defined"
#endif

int register_Emulator$Config(JNIEnv* env){

    static const JNINativeMethod methods[] = {

            { "native_open_config", "(Ljava/lang/String;)J", (void *) j_open_config_str },
            { "native_close_config", "(J)Ljava/lang/String;", (void *) j_close_config_str },
            { "native_open_config_file", "(Ljava/lang/String;)J", (void *) j_open_config_file },
            { "native_load_config_entry", "(JLjava/lang/String;)Ljava/lang/String;", (void *) j_load_config_entry },
            { "native_load_config_entry_ty_arr", "(JLjava/lang/String;)[Ljava/lang/String;", (void *) j_load_config_entry_ty_arr },
            { "native_save_config_entry", "(JLjava/lang/String;Ljava/lang/String;)V", (void *) j_save_config_entry },
            { "native_save_config_entry_ty_arr", "(JLjava/lang/String;[Ljava/lang/String;)V", (void *) j_save_config_entry_ty_arr },
            { "native_close_config_file", "(JLjava/lang/String;)V", (void *) j_close_config_file },
            { "native_free_config", "(J)V", (void *) j_free_config },
    };
    jclass clazz = env->FindClass("xendroid/emulator/Emulator$Config");
    return env->RegisterNatives(clazz,methods, sizeof(methods)/sizeof(methods[0]));
}

//public native void setup_game_path(String path);
//public native void setup_game_path(Emulator.Path path);
static void j_setup_game_path(JNIEnv* env,jobject self,jobject path ){

    jclass path_cls;
    if(env->IsInstanceOf(path,path_cls=env->FindClass("xendroid/emulator/Emulator$Path"))){
        jfieldID f_fd  = env->GetFieldID(path_cls, "fd", "I");
        ae::boot_game_fd=env->GetIntField(path, f_fd);
        ae::boot_type=ae::BOOT_TYPE_WITH_FD;
    }
    else if(env->IsInstanceOf(path,path_cls=env->FindClass("java/lang/String"))){
        const char*  _path = env->GetStringUTFChars(reinterpret_cast<jstring>(path), nullptr);
        ae::boot_type=ae::BOOT_TYPE_WITH_PATH;
        ae::boot_game_path=std::string(_path);
        env->ReleaseStringUTFChars(reinterpret_cast<jstring>(path), _path);
    }
}

static void j_boot(JNIEnv* env,jobject self){
    std::thread(ae::main_thr).detach();
}

static void j_change_surface(JNIEnv* env,jobject self,jint w,jint h){
    ae::surface_resize(w,h);
}

static void j_setup_surface(JNIEnv* env,jobject self,jobject surface){
    if(surface){
        // ANativeWindow_fromSurface returns one owned reference; ae::surface_*
        // owns it from here and releases it on detach.
        ANativeWindow* w = ANativeWindow_fromSurface(env,surface);
        ae::surface_attach(w, ANativeWindow_getWidth(w), ANativeWindow_getHeight(w));
    } else {
        ae::surface_detach();
    }
}

static void j_key_event(JNIEnv* env,jobject self,jint key_code,jboolean pressed,jint value){
    ae::key_event(key_code,pressed,value);
}

static void j_pause(JNIEnv* env,jobject self){
    ae::pause();
}

static void j_resume(JNIEnv* env,jobject self){
    ae::resume();
}

static jboolean j_is_running(JNIEnv* env,jobject self){
    return ae::is_running();
}

static jboolean j_is_paused(JNIEnv* env,jobject self){
    return ae::is_paused();
}

static void j_quit(JNIEnv* env,jobject self){
    ae::quit();
}

static void j_flush_gpu_caches(JNIEnv* env,jobject self){
    ae::flush_gpu_caches();
}

int register_Emulator(JNIEnv* env){
    static const JNINativeMethod methods[] = {
            { "setup_game_path", "(Ljava/lang/String;)V", (void *) j_setup_game_path },
            { "setup_game_path", "(Lxendroid/emulator/Emulator$Path;)V", (void *) j_setup_game_path },
            { "setup_surface", "(Landroid/view/Surface;)V", (void *) j_setup_surface },
            { "boot", "()V", (void *) j_boot },
            { "key_event", "(IZI)V", (void *) j_key_event },
            { "quit", "()V", (void *) j_quit },
            { "is_running", "()Z", (void *) j_is_running },
            { "is_paused", "()Z", (void *) j_is_paused },
            { "pause", "()V", (void *) j_pause },
            { "resume", "()V", (void *) j_resume },
            { "change_surface", "(II)V", (void *) j_change_surface },
            { "flush_gpu_caches", "()V", (void *) j_flush_gpu_caches },
    };
    return env->RegisterNatives(env->FindClass("xendroid/emulator/Emulator"),methods, sizeof(methods)/sizeof(methods[0]));
}