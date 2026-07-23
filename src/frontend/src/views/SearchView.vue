<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { searchKnowledge, type SearchResult } from '@/api/knowledge'

const router = useRouter()
const query = ref('')
const topK = ref(5)
const isSearching = ref(false)
const searchResults = ref<SearchResult[]>([])
const errorMessage = ref('')
const hasSearched = ref(false)

const goToUpload = () => {
  router.push('/upload')
}

const handleSearch = async () => {
  if (!query.value.trim()) {
    errorMessage.value = '请输入检索内容'
    return
  }

  isSearching.value = true
  errorMessage.value = ''
  hasSearched.value = true
  searchResults.value = []

  try {
    const response = await searchKnowledge(query.value, topK.value)
    searchResults.value = response.results
  } catch (error: any) {
    console.error('Search error:', error)
    errorMessage.value = error.response?.data?.detail || '检索失败，请重试'
  } finally {
    isSearching.value = false
  }
}

const formatScore = (score: number): string => {
  return (score * 100).toFixed(1)
}

const highlightText = (text: string): string => {
  if (!query.value.trim()) return text
  const keywords = query.value.split(/\s+/).filter(k => k.length > 0)
  let result = text
  keywords.forEach(keyword => {
    const regex = new RegExp(`(${keyword})`, 'gi')
    result = result.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>')
  })
  return result
}
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100">
    <div class="container mx-auto px-4 py-8">
      <!-- Header with Navigation -->
      <div class="max-w-4xl mx-auto mb-8 flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-800 mb-2">知识库检索</h1>
          <p class="text-gray-600">基于语义的智能检索，快速定位所需知识</p>
        </div>
        <button
          @click="goToUpload"
          class="px-4 py-2 bg-white border border-indigo-600 text-indigo-600 rounded-lg hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors font-medium flex items-center gap-2"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          文档上传
        </button>
      </div>

      <!-- Search Box -->
      <div class="max-w-4xl mx-auto mb-8">
        <div class="bg-white rounded-xl shadow-lg p-6">
          <div class="flex flex-col md:flex-row gap-4">
            <div class="flex-1">
              <input
                v-model="query"
                @keyup.enter="handleSearch"
                type="text"
                placeholder="输入您的问题或关键词..."
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-colors"
              />
            </div>
            <div class="flex items-center gap-2">
              <label class="text-sm text-gray-600 whitespace-nowrap">
                结果数量：
              </label>
              <select
                v-model="topK"
                class="px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none bg-white"
              >
                <option :value="3">3 条</option>
                <option :value="5">5 条</option>
                <option :value="10">10 条</option>
                <option :value="20">20 条</option>
              </select>
            </div>
            <button
              @click="handleSearch"
              :disabled="isSearching || !query.trim()"
              class="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {{ isSearching ? '检索中...' : '检索' }}
            </button>
          </div>

          <div v-if="errorMessage" class="mt-4">
            <div class="bg-red-50 border border-red-200 rounded-lg p-3">
              <p class="text-sm text-red-800">{{ errorMessage }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Results -->
      <div class="max-w-4xl mx-auto">
        <div v-if="hasSearched && !isSearching && searchResults.length === 0 && !errorMessage" class="bg-white rounded-xl shadow-lg p-8 text-center">
          <svg class="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p class="text-gray-600">未找到相关结果，请尝试其他关键词</p>
        </div>

        <div v-if="searchResults.length > 0" class="space-y-4">
          <div class="mb-4 text-sm text-gray-600">
            找到 <span class="font-semibold text-indigo-600">{{ searchResults.length }}</span> 条相关结果
          </div>

          <div
            v-for="(result, index) in searchResults"
            :key="index"
            class="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow"
          >
            <div class="flex items-start justify-between mb-3">
              <div class="flex items-center gap-3">
                <span class="flex items-center justify-center w-8 h-8 bg-indigo-100 text-indigo-600 rounded-full font-semibold text-sm">
                  {{ index + 1 }}
                </span>
                <div class="flex items-center gap-2">
                  <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    匹配度：{{ formatScore(result.score) }}%
                  </span>
                </div>
              </div>
            </div>

            <div
              class="text-gray-700 leading-relaxed"
              v-html="highlightText(result.content)"
            ></div>

            <div v-if="result.doc_name || result.headers" class="mt-4 pt-4 border-t border-gray-100">
              <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-500">
                <div v-if="result.doc_name">
                  <span class="font-medium">文档:</span> {{ result.doc_name }}
                </div>
                <div v-if="result.headers">
                  <span class="font-medium">章节:</span> {{ result.headers }}
                </div>
                <div>
                  <span class="font-medium">信任值:</span> {{ (result.trust_score * 100).toFixed(1) }}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
mark {
  padding: 0.1em 0.2em;
  border-radius: 0.2em;
}
</style>
